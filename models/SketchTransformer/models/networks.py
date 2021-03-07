import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import math
import numpy as np
import torchvision.models as models

from modules.networks import get_pad
from torch.distributions.multivariate_normal import MultivariateNormal
from util.utils import length_to_mask

def get_conv_layer(in_channel, out_channel, gan_type='sn_gan',  **kwargs):
    if gan_type == 'sn_gan':
        return spectral_norm(nn.Conv2d(in_channel, out_channel, **kwargs))
    else:
        return nn.Conv2d(in_channel, out_channel, **kwargs)

def get_conv_block(in_channel, out_channel, gan_type='sn_gan', normalization='instance', activation='leakyrelu', **kwargs):
    block = []
    block.append(get_conv_layer(in_channel, out_channel, gan_type=gan_type, **kwargs))
    if normalization == 'instance':
        block.append(nn.InstanceNorm2d(out_channel))

    if activation == 'leakyrelu':
        block.append(nn.LeakyReLU())
    return nn.Sequential(*block)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as SketchLayerNorm
# except ImportError:
#     logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
class SketchLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(SketchLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}#, "swish": swish
NORM2FN = {'BN1d':nn.BatchNorm1d, 'BN2d':nn.BatchNorm2d, 'LN':nn.LayerNorm}

class SketchSelfAttention(nn.Module):
    '''
    Implementation for self attention in Sketch.
    The input will be a K-Dim feature.
    Input Parameters:
        config[dict]:
            hidden_dim[int]: The dimension of input hidden embeddings in the self attention, hidden diension is equal to the output dimension
            num_heads[int]: The number of heads
            attention_probs[float]: probability parameter for dropout
    '''
    def __init__(self, num_heads, hidden_dim, attention_dropout_prob):
        super(SketchSelfAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        #self.attention_dropout_prob = config.attention_dropout_prob
        # Calculation for intermeidate parameters
        self.head_dim = int(self.hidden_dim / self.num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale_factor = math.sqrt(self.head_dim)

        self.query = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.key = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.value = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.dropout = nn.Dropout(attention_dropout_prob)
        self.multihead_output = None

    def transpose_(self, x):
        '''
        Transpose Function for simplicity.
        '''
        new_x_shape = x.size()[:-1] + (self.num_heads , self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False, keep_multihead_output=False):
        '''
        Input:
            hidden_states[batch, seq_len, hidden_dim]
            attention_mask[batch,  1, 1, seq_len]
        Output:
            context_states[batch, seq_len, hidden_dim]
            attention_probs[seq_len, hidden_dim]
        '''
        # Get query, key, value together
        query = self.query(hidden_states) # [batch, seq_len, all_head_dim]
        key = self.key(hidden_states) # [batch, seq_len, all_head_dim]
        value = self.value(hidden_states) # [batch, seq_len, all_head_dim]

        # tranpose the query, key, value into multi heads[batch, seq_len, ]
        multi_query = self.transpose_(query) # [batch, num_heads, seq_len, head_dim]
        multi_key = self.transpose_(key) # [batch,  num_heads, seq_len, head_dim]
        multi_value = self.transpose_(value) # [batch, num_heads, seq_len, head_dim]

        # Calculate Attention maps
        attention_scores = torch.matmul(multi_query, multi_key.transpose(-1, -2))
        attention_scores = attention_scores / self.scale_factor
        #print(attention_scores.size(), attention_mask.size())
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # Compute states values
        context_states = torch.matmul(attention_probs, multi_value)

        if keep_multihead_output:
            self.multihead_output = context_states
            self.multihead_output.retain_grad()

        context_states = context_states.permute(0,2,1,3)
        context_states = context_states.contiguous().view(context_states.size()[:-2]+(-1,)) #view(context_states.size()[:-2]+ (self.all_head_dim,))

        if output_attentions:
            return context_states, attention_probs
        return context_states


class SketchOutput(nn.Module):
    def __init__(self, input_dim, output_dim, attention_norm_type, output_dropout_prob):
        super(SketchOutput, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if attention_norm_type not in NORM2FN:
            raise ValueError(
                "The attention normalization is not in standard normalization types.")
        self.norm = NORM2FN[attention_norm_type](output_dim)
        self.dropout = nn.Dropout(output_dropout_prob)
    '''
    Input:
        hidden_states[]:

    Output:
        hidden_states[]:
    '''
    def forward(self, hidden_states, input_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #print(hidden_states.size())
        hidden_states = self.norm(hidden_states+input_states)
        return hidden_states


class SketchMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim,
                 attention_norm_type, attention_dropout_prob, hidden_dropout_prob,):
        super(SketchMultiHeadAttention, self).__init__()
        self.attention = SketchSelfAttention(num_heads, hidden_dim, attention_dropout_prob)
        self.output = SketchOutput(hidden_dim, hidden_dim, attention_norm_type, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):
        input_states = hidden_states
        #print(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, head_mask=head_mask)
        #print(hidden_states)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        output_states = self.output(hidden_states, input_states)
        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchIntermediate(nn.Module):
    def __init__(self, hidden_dim, inter_dim, inter_activation):
        super(SketchIntermediate, self).__init__()
        self.fc = nn.Linear(hidden_dim, inter_dim)
        self.activation = ACT2FN[inter_activation]


    def forward(self, hidden_states):

        hidden_states = hidden_states.to(next(self.fc.parameters()).device)

        inter_states = self.fc(hidden_states.contiguous())
        inter_states = self.activation(inter_states)
        return inter_states

class SketchLayer(nn.Module):
    '''
        A transformer layer for sketch bert
    '''
    def __init__(self, num_heads, hidden_dim, inter_dim,
                 attention_norm_type, inter_activation, attention_dropout_prob,
                 hidden_dropout_prob, output_dropout_prob,):
        super(SketchLayer, self).__init__()
        self.attention = SketchMultiHeadAttention(num_heads, hidden_dim,
                                                    attention_norm_type, attention_dropout_prob, hidden_dropout_prob,)
        self.inter_layer = SketchIntermediate(hidden_dim, inter_dim, inter_activation)
        self.output = SketchOutput(inter_dim, hidden_dim, attention_norm_type, output_dropout_prob)


    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]:
        attention_mask[batch, seq_len]


    '''
    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):

        hidden_states = self.attention(hidden_states, attention_mask, head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        inter_states = self.inter_layer(hidden_states)
        output_states = self.output(inter_states, hidden_states)

        if output_attentions:
            return output_states, attention_probs

        return output_states

class SketchSegmentLayer(nn.Module):
    '''
        A transformer layer for sketch bert
    '''
    def __init__(self, num_heads, hidden_dim, inter_dim, max_segment,
                 segment_atten_type, attention_norm_type, inter_activation, attention_dropout_prob,
                 hidden_dropout_prob, output_dropout_prob,):
        super(SketchSegmentLayer, self).__init__()
        self.max_segment = max_segment
        self.inter_dim = inter_dim
        self.segment_atten_type = segment_atten_type
        self.local_attention = SketchMultiHeadAttention(num_heads, hidden_dim,
                                                    attention_norm_type, attention_dropout_prob, hidden_dropout_prob,)
        self.segment_attention = SketchMultiHeadAttention(num_heads, hidden_dim,
                                                    attention_norm_type, attention_dropout_prob, hidden_dropout_prob,)
        self.local_inter_layer = SketchIntermediate(hidden_dim, inter_dim//2, inter_activation)
        self.seg_inter_layer = SketchIntermediate(hidden_dim, inter_dim//2, inter_activation)
        self.output = SketchOutput(inter_dim, hidden_dim, attention_norm_type, output_dropout_prob)


    def get_seg_states(self, hidden_states, segment_index):
        '''
        Input:
            hidden_states[batch, seq_len, hidden_dim]
            segment_index[batch, seq_len]
        '''
        seg_states = torch.zeros(hidden_states.size(0), self.max_segment, hidden_states.size(2)).to(hidden_states.device)
        length = (segment_index==0).sum(dim=1)
        length_mask = length_to_mask(length, max_len=self.max_segment, dtype=torch.float)
        seg_states[length_mask==1,:] = hidden_states[segment_index==0,:]
        return seg_states, length_mask

    def forward(self, hidden_states, attention_mask, segments, segment_index, head_mask=None, output_attentions=False):
        '''
        Input:
            hidden_states[batch, seg_len, hidden_dim]:
            attention_mask[batch, seg_len](segment-based)
            segments[batch, seg_len]:
            segment_index[batch, seq_len]

        '''
        # Local Attention
        local_states = self.local_attention(hidden_states, attention_mask, head_mask)
        if output_attentions:
            local_states, attention_probs = local_states #[batch, seq_len, hidden_dim]
        input_prefix = hidden_states.size(1) - segment_index.size(1)

        # Segment Level Attention
        seg_states, seg_atten_mask = self.get_seg_states(local_states[:,input_prefix:,:], segment_index)
        if self.segment_atten_type == 'multi':
            seg_states = self.segment_attention(seg_states, seg_atten_mask.unsqueeze(1).unsqueeze(2), head_mask)
            if output_attentions:
                seg_states, attention_probs = seg_states #[batch, seq_len, hidden_dim]

        # Concatenate
        local_inter_states = self.local_inter_layer(local_states)
        seg_inter_states = self.seg_inter_layer(seg_states)
        aug_seg_inter_states = torch.gather(seg_inter_states, 1, (segments[:,input_prefix:]-2).view(segments.size(0), -1, 1).repeat(1,1, seg_inter_states.size(2)))
        inter_states = torch.zeros(local_inter_states.size(0), local_inter_states.size(1), self.inter_dim).to(local_inter_states.device)
        #print(hidden_states.size(), local_states.size(), local_inter_states.size())
        inter_states[:,:,:self.inter_dim//2] = local_inter_states
        inter_states[:,input_prefix:, self.inter_dim//2:] = aug_seg_inter_states
        inter_states[:,:input_prefix, self.inter_dim//2:] = seg_inter_states.sum(dim=1, keepdim=True)

        output_states = self.output(inter_states, hidden_states)

        if output_attentions:
            return output_states, attention_probs

        return output_states


def setting2dict(paras, setting):
    paras['num_heads'] = setting[0]
    paras['hidden_dim'] = setting[1]
    paras['inter_dim'] = setting[2]

class SketchEncoder(nn.Module):
    '''
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        self.layers = []
        for layer_setting in layers_setting:
            setting2dict(layer_paras, layer_setting)
            self.layers.append(SketchLayer(**layer_paras))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchALEncoder(nn.Module):
    '''
        A Lite BERT: Parameter Sharing
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchALEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        setting2dict(layer_paras, layers_setting[0])
        self.sketch_layer = SketchLayer(**layer_paras)
        self.layers = []
        for layer_setting in layers_setting:
            self.layers.append(self.sketch_layer)
        #self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states

class SketchSegmentEncoder(nn.Module):
    '''
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting, max_segment, segment_atten_type,
                    attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchSegmentEncoder, self).__init__()
        layer_paras = {
                     'max_segment':max_segment, 'segment_atten_type':segment_atten_type, 'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        self.layers = []
        self.max_segment = max_segment
        for layer_setting in layers_setting:
            setting2dict(layer_paras, layer_setting)
            self.layers.append(SketchSegmentLayer(**layer_paras))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, segments, segment_index, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, segments, segment_index, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states

class SketchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SketchEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_states):
        return self.embedding(input_states)

class SketchDiscreteEmbedding(nn.Module):
    '''
    max_size[tuple](x_length, y_length)
    '''
    def __init__(self, max_size, type_size, hidden_dim, pool_type):
        super(SketchDiscreteEmbedding, self).__init__()
        self.x_embedding = nn.Embedding(2*max_size[0]+2, hidden_dim//2)
        self.y_embedding = nn.Embedding(2*max_size[1]+2, hidden_dim//2)
        self.type_embedding = nn.Embedding(type_size+1, hidden_dim)
        assert pool_type in ['sum', 'con']
        self.pool_type = pool_type

    '''
    input_states[batch, seq_len, 3(input_dim)](Inputs are encoded as discrete type)
    '''
    def forward(self, input_states):
        input_states = input_states.to(dtype=torch.long)
        input_states = input_states + 1
        #print(input_states[0,0,:], torch.min(input_states), torch.max(input_states))
        x_hidden = self.x_embedding(input_states[:,:,0])
        y_hidden = self.y_embedding(input_states[:,:,1])
        #print(x_hidden.size(), y_hidden.size())
        axis_hidden = torch.cat([x_hidden, y_hidden], dim=2)

        type_hidden = self.type_embedding(input_states[:,:,2])

        if self.pool_type == 'sum':
            return axis_hidden + type_hidden
        elif self.pool_type == 'con':
            return torch.cat([axis_hidden, type_hidden], dim=2)

class SketchSinPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchSinPositionEmbedding, self).__init__()
        self.pos_embedding_matrix = torch.zeros(max_length, pos_hidden_dim)
        pos_vector = torch.arange(max_length).view(max_length, 1).type(torch.float)
        dim_vector = torch.arange(pos_hidden_dim).type(torch.float) + 1.0
        #print((pos_vector / (dim_vector[::2] / 2).view(1, -1)).size(), self.pos_embedding_matrix[:,::2].size())
        self.pos_embedding_matrix[:,::2] = torch.sin(pos_vector / (dim_vector[::2] / 2).view(1, -1))
        self.pos_embedding_matrix[:,1::2] = torch.cos(pos_vector / ((dim_vector[1::2] - 1) / 2).view(1, -1))
        #print(self.pos_embedding_matrix)
    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding_matrix[position_labels.view(-1),:].view(position_labels.size(0), position_labels.size(1), -1)

class SketchLearnPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchLearnPositionEmbedding, self).__init__()
        print(max_length, pos_hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, pos_hidden_dim)

    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding(position_labels)

class SketchEmbeddingRefineNetwork(nn.Module):
    '''
    The module to upsample the embedding feature, idea from the ALBERT: Factorized Embedding
    '''
    def __init__(self, out_dim, layers_dim):
        super(SketchEmbeddingRefineNetwork, self).__init__()
        self.layers = []
        layers_dim = layers_dim.copy()
        layers_dim.append(out_dim)

        for i in range(len(layers_dim)-1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_state):
        x = input_state
        for layer in self.layers:
            x = layer(x)
        return x

class SketchTransformerModel(nn.Module):
    '''
    Input:
        layers_setting[list]
        input_dim[int]
        max_length[int]
        position_type[str]
        attention_norm_type[str]
        inter_activation[str]
        attention_dropout_prob[float]
        hidden_dropout_prob[float]
        output_dropout_prob[float]
    '''
    def __init__(self, model_type, layers_setting, embed_layers_setting, input_dim, max_length, max_size, type_size,
                    position_type, segment_type, sketch_embed_type, embed_pool_type, attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob):
        super(SketchTransformerModel, self).__init__()
        self.layers_setting = layers_setting
        self.num_hidden_layers = len(layers_setting)
        self.embed_pool_type = embed_pool_type
        assert sketch_embed_type in ['linear', 'discrete']

        if sketch_embed_type == 'linear':
            self.embedding = SketchEmbedding(input_dim, embed_layers_setting[0])
        elif sketch_embed_type == 'discrete':
            self.embedding = SketchDiscreteEmbedding(max_size, type_size, embed_layers_setting[0], embed_pool_type)
        assert position_type in ['sin', 'learn', 'none']

        if position_type == 'sin':
            self.pos_embedding = SketchSinPositionEmbedding(max_length, embed_layers_setting[0])
        elif position_type == 'learn':
            self.pos_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.pos_embedding = None
        if segment_type == 'learn':
            self.segment_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.segment_embedding = None

        self.embed_refine_net = SketchEmbeddingRefineNetwork(layers_setting[0][1], embed_layers_setting)

        assert model_type in ['albert', 'bert']
        if model_type == 'albert':
            self.encoder = SketchALEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)
        elif model_type == 'bert':
            self.encoder = SketchEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)

    def load_model(self, state_dict, own_rel_in_input, own_cls_in_input, pre_rel_in_input, pre_cls_in_input):
        own_state = self.state_dict()
        for k, v in own_state.items():
            if k == 'pos_embedding.pos_embedding.weight':
                own_pos_size = v.size(0)
                seq_len = own_pos_size - own_rel_in_input - own_cls_in_input
                pretrained_pos_size = state_dict[k].size(0)
                own_start_ind = int(own_rel_in_input+own_cls_in_input)
                pre_start_ind = int(pre_rel_in_input+pre_cls_in_input)
                seq_len = min(seq_len, state_dict[k].size(0)-pre_start_ind)
                own_state[k][own_start_ind:own_start_ind+seq_len] = state_dict[k][pre_start_ind:pre_start_ind+seq_len]
                if own_rel_in_input and own_cls_in_input:
                    if pre_cls_in_input and pre_cls_in_input:
                        own_state[k][:2] = state_dict[k][:2]
                    elif pre_cls_in_input:
                        own_state[k][1] = state_dict[k][0]
                    elif pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_rel_in_input:
                    if pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_cls_in_input:
                    if pre_cls_in_input:
                        own_state[k][0] = state_dict[k][int(pre_rel_in_input)]
            else:
                own_state[k] = state_dict[k]
        self.load_state_dict(own_state)

    def get_pos_states(self, input_states):
        return torch.arange(input_states.size(1)).view(1,-1).repeat(input_states.size(0),1).to(device=input_states.device)
    '''
    Input:
        input_states[batch, seq_len, 5],
        attention_mask[batch, seq_len]/[batch, seq_len, ],(length mask)
    Output:
        output_states[batch, seq_len, hidden_dim],
    '''
    def forward(self, input_states, attention_mask, segments=None, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))
        # Extending attention mask
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask
        # process head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # switch to fload if need + fp16 compatibility
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states.to(device=input_states.device)

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states
        input_states = self.embed_refine_net(input_states)
        output_states = self.encoder(input_states, attention_mask, head_mask, output_all_states, output_attentions, keep_multihead_output)


        if output_attentions:
            output_states, attention_probs = output_states
            return output_states[-1], attention_probs

        return output_states[-1]

class SketchCNN(nn.Module):
    '''
    Truely a CNN model
    '''
    def __init__(self, hidden_dim, net_type, pretrained):
        super(SketchCNN, self).__init__()
        if net_type == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, hidden_dim)
        elif net_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, hidden_dim)
        elif net_type == 'tcnet':
            pass
        elif net_type =='sketchanet':
            pass

    def forward(self, input):
        return self.encoder(input)


'''
Sketch Transformer based GAN
'''
class SketchGANGenerator(nn.Module):
    '''
    Assume Label in the Input
    '''
    def __init__(self, layers_setting, input_dim, cls_dim, max_length,
                    position_type, attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob):
        super(SketchGANGenerator, self).__init__()
        self.encoder = SketchTransformerModel(layers_setting, input_dim, cls_dim,  max_length,
                        position_type, attention_norm_type, inter_activation, attention_dropout_prob,
                        hidden_dropout_prob, output_dropout_prob)
        self.output = nn.Linear(layers_setting[0][1], 5)

    '''
    The same as Transformer Model
    '''
    def forward(self, input_states, attention_mask, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        hidden_states = self.encoder(input_states, attention_mask, head_mask=head_mask,
                    output_all_states=output_all_states, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        fake_states = self.output(hidden_states)
        return fake_states

class SketchGANDiscriminator(nn.Module):
    '''
    Assume Label in the Input
    '''
    def __init__(self, layers_setting, input_dim, cls_dim, max_length,
                    position_type, attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob):
        super(SketchGANDiscriminator, self).__init__()
        self.encoder = SketchTransformerModel(layers_setting, input_dim, cls_dim,  max_length,
                        position_type, attention_norm_type, inter_activation, attention_dropout_prob,
                        hidden_dropout_prob, output_dropout_prob)
        self.output = nn.Linear(layers_setting[0][1], 2)

    '''
    The same as Transformer Model
    '''
    def forward(self, input_states, attention_mask, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        hidden_states = self.encoder(input_states, attention_mask, head_mask=head_mask,
                    output_all_states=output_all_states, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        label = self.output(hidden_states[:,0,:])
        return label


'''
Sketch Transformer based VAE
'''
class SketchVAEEncoder(SketchTransformerModel):
    def __init__(self, model_type, layers_setting, embed_layers_setting, input_dim, cls_dim, max_length, max_size, type_size,
                    conditional, position_type, segment_type, sketch_embed_type, embed_pool_type, attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob):
        super(SketchVAEEncoder, self).__init__(model_type, layers_setting, embed_layers_setting, input_dim, max_length, max_size, type_size,
                        position_type, segment_type, sketch_embed_type, embed_pool_type, attention_norm_type, inter_activation, attention_dropout_prob,
                        hidden_dropout_prob, output_dropout_prob)
        # self.rec_fc = nn.Linear(layers_setting[0][1], output_dim)
        self.conditional = conditional
        if self.conditional:
            self.cls_embedding = nn.Embedding(cls_dim, embed_layers_setting[0])
        else:
            self.cls_embedding = None
    def load_model(self, state_dict, only_encoder):
        own_state = self.state_dict()
        for k, v in own_state.items():
            if only_encoder and ('encoder' in k or 'embed_refine_net' in k):
                own_state[k] = state_dict[k]
            else:
                if k in state_dict and k in own_state:
                    own_state[k] = state_dict[k]
        self.load_state_dict(own_state)
    def forward(self, input_states, attention_mask, targets=None, segments=None, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        '''
        Input:
            input_states[batch, seq_len, 5],
            zs[batch, latent_dim]
        '''
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))
        # Extending attention mask
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        # process head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # switch to fload if need + fp16 compatibility
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states

        if self.cls_embedding is not None and targets is not None:
            cls_states = self.cls_embedding(targets)
            cls_states = cls_states.unsqueeze(1).repeat(1,input_states.size(1),1)
            input_states = input_states + cls_states
        input_states = self.embed_refine_net(input_states)
        # Append the latent_states
        output_states = self.encoder(input_states, attention_mask, head_mask, output_all_states, output_attentions, keep_multihead_output)

        if output_attentions:
            output_states, attention_probs = output_states
            #return self.rec_fc(output_states[-1]), attention_probs
            return output_states[-1], attention_probs
        return output_states[-1]

'''
Sketch Transformer based VAE
'''
class SketchVAEDecoder(SketchTransformerModel):

    def __init__(self, model_type, layers_setting, embed_layers_setting, rec_layers_setting, input_dim, output_dim, latent_dim, cls_dim, max_length, max_size, type_size,
                    conditional, position_type, segment_type, sketch_embed_type, embed_pool_type, attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob):
        print(embed_layers_setting)
        super(SketchVAEDecoder, self).__init__(model_type, layers_setting, embed_layers_setting, input_dim, max_length, max_size, type_size,
                        position_type, segment_type, sketch_embed_type, embed_pool_type, attention_norm_type, inter_activation, attention_dropout_prob,
                        hidden_dropout_prob, output_dropout_prob)
        self.conditional = conditional
        if self.conditional:
            self.cls_embedding = nn.Embedding(cls_dim, embed_layers_setting[0])
        else:
            self.cls_embedding = None
        self.re_fcs = []
        rec_layers_setting = rec_layers_setting.copy()
        rec_layers_setting.append(output_dim), rec_layers_setting.insert(0, layers_setting[0][1])
        for i in range(len(rec_layers_setting)-1):
            self.re_fcs.append(nn.Linear(rec_layers_setting[i], rec_layers_setting[i+1]))
        self.re_fcs = nn.ModuleList(self.re_fcs)

        self.latent_fusion = nn.Linear(layers_setting[0][1]+latent_dim, layers_setting[0][1])
    def load_model(self, state_dict, only_encoder):
        own_state = self.state_dict()
        for k, v in own_state.items():
            if only_encoder and ('encoder' in k or 'embed_refine_net' in k):
                #print(k in own_state, k in state_dict)
                own_state[k] = state_dict[k]
            else:
                if k in state_dict and k in own_state:
                    own_state[k] = state_dict[k]
        self.load_state_dict(own_state)
    def forward(self, input_states, zs, attention_mask, targets=None, segments=None, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        '''
        Input:
            input_states[batch, seq_len, 5],
            zs[batch, latent_dim]
        '''
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))

        # Extending attention mask
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        # process head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # switch to fload if need + fp16 compatibility
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states

        if self.cls_embedding is not None and targets is not None:
            cls_states = self.cls_embedding(targets)
            cls_states = cls_states.unsqueeze(1).repeat(1,input_states.size(1),1)
            input_states = input_states + cls_states

        input_states = self.embed_refine_net(input_states)
        # Append the latent_states
        input_states = torch.cat([input_states, zs.unsqueeze(1).repeat(1,input_states.size(1),1)],dim=2)
        input_states = self.latent_fusion(input_states)
        output_states = self.encoder(input_states, attention_mask, head_mask, output_all_states, output_attentions, keep_multihead_output)

        if output_attentions:
            output_states, attention_probs = output_states
            output_states = output_states[-1]
            for re_fc in self.re_fcs:
                output_states = re_fc(output_states)
            return output_states, attention_probs
        output_states = output_states[-1]
        for re_fc in self.re_fcs:
            output_states = re_fc(output_states)
        return output_states


class SketchVAELatentEmbedding(nn.Module):
    def __init__(self, hidden_dim, latent_dim, max_length):
        super(SketchVAELatentEmbedding, self).__init__()
        self.mu_embedding = nn.Linear(hidden_dim, latent_dim)
        self.sigma_embedding = nn.Linear(hidden_dim, latent_dim)
        self.gaussian_generator = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]
    Output:
        mus[batch, latent_dim]
        sigmas[batch, latent_dim]
        z[batch, latent_dim]
    '''
    def forward(self, hidden_states, attention_mask):

        # Mask the lengths beyond
        latent_states = hidden_states[:,0,:]
        mus = self.mu_embedding(latent_states)
        sigmas = self.sigma_embedding(latent_states)
        sigmas = torch.exp(sigmas/2)
        random_normal = self.gaussian_generator.sample([sigmas.size(0)]).to(sigmas.device)
        zs = mus + sigmas * random_normal
        return mus, sigmas , zs



'''
Different Pooling Layers
'''
class SketchPooling(nn.Module):
    def __init__(self, hidden_dim, input_dim, cls_dim, max_length=250):
        super(SketchPooling, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 4)
        self.fc2 = nn.Linear(max_length*4, cls_dim)
        self.re_fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, hidden_states):
        re_sketch = self.re_fc(hidden_states)
        pooled = self.fc1(hidden_states)
        pooled = self.fc2(pooled.view(pooled.size(0), -1))
        return re_sketch, pooled

class SketchGMMPooling(nn.Module):
    def __init__(self, hidden_dim, M, cls_dim, max_length=250):
        super(SketchPooling, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 4)
        self.fc2 = nn.Linear(max_length*4, cls_dim)
        self.re_fc = nn.Linear(hidden_dim, 6*M + 3)

    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]
    Output:
        re_sketch[batch, seq_len, 6M+3]
        pooled[batch, cls_dim]
    '''
    def forward(self, hidden_states):
        re_sketch = self.re_fc(hidden_states)
        pooled = self.fc1(hidden_states)
        pooled = self.fc2(pooled.view(pooled.size(0), -1))
        return re_sketch, pooled


class SketchHiddenPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(SketchPooler, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.fc(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

'''
Multi models for transformer backbone
'''

#Mask Sketch Model
class MaskSketchRecModel(nn.Module):
    def __init__(self, rec_layers_setting, hidden_dim, input_dim, cls_in_input, rel_in_input):
        super(MaskSketchRecModel, self).__init__()
        self.re_fcs = []
        rec_layers_setting.append(input_dim), rec_layers_setting.insert(0, hidden_dim)
        for i in range(len(rec_layers_setting)-1):
            self.re_fcs.append(nn.Linear(rec_layers_setting[i], rec_layers_setting[i+1]))
        self.re_fcs = nn.ModuleList(self.re_fcs)

        self.cls_in_input = cls_in_input
        self.rel_in_input = rel_in_input
    '''
    Input:
        hidden_states[batch, seq_len+cls_input, hidden_dim]

    '''
    def forward(self, hidden_states):
        hidden_states = hidden_states[:, self.cls_in_input+self.rel_in_input:, :]
        for re_fc in self.re_fcs:
            hidden_states = re_fc(hidden_states)
        return hidden_states

class MaskSketchGMMModel(nn.Module):
    def __init__(self, hidden_dim, M, cls_in_input, rel_in_input):
        super(MaskSketchGMMModel, self).__init__()
        self.re_fc = nn.Linear(hidden_dim, 6*M + 3)
        self.cls_in_input = cls_in_input
        self.rel_in_input = rel_in_input
    '''
    Input:
        hidden_states[batch, seq_len+cls_input, hidden_dim]
        attention_mask[batch, seq_len+cls_input]
    '''
    def forward(self, hidden_states):

        hidden_states = hidden_states[:, self.cls_in_input+self.rel_in_input:, :]
        return self.re_fc(hidden_states)

# Sketch classification model
class SketchClassificationModel(nn.Module):
    def __init__(self, hidden_dim, cls_dim, max_length):
        super(SketchClassificationModel, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 4)
        self.fc2 = nn.Linear(max_length*4, cls_dim)
    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]
        attention_mask[batch, seq_len]
    Output:
        cls_states[batch, cls_dim]
    '''
    def forward(self, hidden_states):
        pooled = self.fc1(hidden_states)
        pooled = self.fc2(pooled.view(pooled.size(0), -1))
        return pooled

class SketchClsPoolingModel(nn.Module):
    def __init__(self, cls_layers_setting, hidden_dim, cls_dim, pool_dim):
        super(SketchClsPoolingModel, self).__init__()
        self.pool_dim = int(pool_dim)
        cls_layers_setting = cls_layers_setting.copy()
        cls_layers_setting.insert(0, hidden_dim), cls_layers_setting.append(cls_dim)
        self.cls_fcs = []
        for i in range(len(cls_layers_setting)-1):
            self.cls_fcs.append(nn.Linear(cls_layers_setting[i], cls_layers_setting[i+1]))
        self.cls_fcs = nn.ModuleList(self.cls_fcs)

    '''
    Input:
        hidden_states[batch, seq_len+cls_dim, hidden_dim](0 dim is cls)
    Output:
        cls_states[batch, cls_dim]
    '''
    def forward(self, hidden_states):
        pooled = hidden_states[:,self.pool_dim,:]
        for cls_fc in self.cls_fcs:
            pooled = cls_fc(pooled)
        return pooled

class SketchRetrievalPoolingModel(nn.Module):
    def __init__(self, rel_layers_setting, hidden_dim, feat_dim, pool_dim):
        super(SketchRetrievalPoolingModel, self).__init__()
        self.pool_dim = int(pool_dim)
        rel_layers_setting = rel_layers_setting.copy()
        rel_layers_setting.insert(0, hidden_dim), rel_layers_setting.append(feat_dim)
        self.rel_fcs = []
        for i in range(len(rel_layers_setting)-1):
            self.rel_fcs.append(nn.Linear(rel_layers_setting[i], rel_layers_setting[i+1]))
        self.rel_fcs = nn.ModuleList(self.rel_fcs)

    '''
    Input:
        hidden_states[batch, seq_len+cls_dim, hidden_dim](0 dim is cls)
    Output:
        cls_states[batch, cls_dim]
    '''
    def forward(self, hidden_states):
        pooled = hidden_states[:,self.pool_dim,:]
        for rel_fc in self.rel_fcs:
            pooled = rel_fc(pooled)
        return pooled

class SketchDiscretePoolingModel(nn.Module):
    def __init__(self, hidden_dim, max_size, type_size, cls_in_input, rel_in_input):
        super(SketchDiscretePoolingModel, self).__init__()
        self.cls_in_input = cls_in_input
        self.rel_in_input = rel_in_input
        self.x_pooling = nn.Linear(hidden_dim, 2*max_size[0]+1)
        self.y_pooling = nn.Linear(hidden_dim, 2*max_size[1]+1)

        self.type_pooling = nn.Linear(hidden_dim, type_size)

    def forward(self, hidden_states):
        '''
        Input:
            hidden_states[batch, seq_len+cls_dim, hidden_dim](0 dim is cls)
        Output:
            x_pred[batch, seq_len+cls_dim, 2*max_size[0]+1]
            y_pred[batch, seq_len+cls_dim, 2*max_size[1]+1]
            type_pred[batch, seq_len+cls_dim, type_size]
        '''
        hidden_states = (hidden_states)[:,self.cls_in_input+self.rel_in_input:,:]
        x_pred = self.x_pooling(hidden_states)
        y_pred = self.y_pooling(hidden_states)
        type_pred = self.type_pooling(hidden_states)
        return x_pred, y_pred, type_pred


class SketchSegmentOrderPoolingModel(nn.Module):
    def __init__(self, hidden_dim, max_segment, cls_in_input, rel_in_input):
        super(SketchSegmentOrderPoolingModel, self).__init__()
        self.sg_fc = nn.Linear(hidden_dim, max_segment)
        self.cls_in_input = cls_in_input
        self.rel_in_input = rel_in_input

    def forward(self, hidden_states, segment_index):
        '''
        Input:
            hidden_states[batch, seg_len, hidden_dim]
            segment_index[batch, seq_len]
        '''
        seg_states = hidden_states[:,self.cls_in_input+self.rel_in_input:,:][segment_index==0,:]
        return self.sg_fc(seg_states)


class GMMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GMMLoss, self).__init__()
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.reduction = reduction

    '''
    x[seq_len, batch, 2]
    lengths[batch, seq_len]
    pis[seq_len, batch, M]: no softmax The {pis} in the paper SketchRNN, https://arxiv.org/abs/1704.03477
    mus[seq_len, batch, M, 2]: The {mus} in the paper SketchRNN, https://arxiv.org/abs/1704.03477
    sigmas[seq_len, batch, M, 2]: exp The {sigmas} in the paper SketchRNN, https://arxiv.org/abs/1704.03477
    rhos[seq_len, batch, M]: tanh The {rho} in the paper SketchRNN, https://arxiv.org/abs/1704.03477
    masks[]
    '''
    def forward(self, x, lengths, pis, mus, sigmas, rhos,  epsilon=1e-8):
        batch_size, seq_len, M = pis.size()
        #print(batch_size, seq_len)
        #print(x.size(), pis.size())
        x = x.view(batch_size, seq_len, 1, 2).repeat(1, 1, M, 1)

        sigma_prods = torch.prod(sigmas, dim=3) # [seq_len, batch, M]
        sigma_sq = torch.pow(sigmas, 2) # [seq_len, batch, M, 2]
        #print(x.size(), mus.size(), sigmas.size())
        x_center = (x - mus) / (sigmas)
        Z = torch.sum(x_center*x_center, dim=3) - 2 * rhos * torch.prod(x_center, dim=3)
        rho_sq = 1 - rhos*rhos  # [seq_len, batch, M]
        denom = 2 * np.pi * sigma_prods * torch.sqrt(rho_sq)
        probs = torch.exp(-Z / (2*rho_sq)) / denom
        # pis = F.softmax(pis, dim=-1)
        probs = torch.sum(F.softmax(pis, dim=-1) * probs, dim=-1)

        log_probs = torch.log(probs+epsilon) * lengths # [len]

        loss = - torch.mean(log_probs)
        return loss


class KLLoss(nn.Module):
    def __init__(self, kl_tolerance):
        super(KLLoss, self).__init__()
        self.kl_tolerance = torch.tensor(kl_tolerance)

    '''
    Input:
        mus[batch, latent_size]:
        sigmas[batch, latent_size]:
    '''
    def forward(self, mus, sigmas):
        loss = - (0.5) * torch.mean(1 + torch.log(sigmas)*2.0  - mus*mus - sigmas*sigmas)
        return torch.max(loss, self.kl_tolerance.to(loss.device))
