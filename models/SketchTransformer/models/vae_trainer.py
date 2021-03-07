import os
import time
import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

from util.evaluation import AverageMeter, accuracy, pairwise_distances
from util.utils import length_to_mask
from trainer.trainer import Trainer
from modules.losses import get_gan_loss, KLLoss
from .networks import SketchVAEEncoder, MaskSketchRecModel, MaskSketchGMMModel, SketchClassificationModel, SketchVAELatentEmbedding, SketchVAEDecoder, GMMLoss, KLLoss
from .utils import strokes2drawing, to_normal_strokes, rec_incomplete_strokes

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

TASKMODELS = {'maskrec':MaskSketchRecModel, 'maskgmm':MaskSketchGMMModel, 'sketchcls':SketchClassificationModel}

class SketchTransformerVAETrainer(Trainer):
    """
    A Trainer for spade, take of the training process and results recording.
    input:
        args[dict]:
        models[dict]:
        train_data[DataLoader, list]:
        val_data[DataLoader, list]:
    """
    def __init__(self, args, datas, log_dir, logger, tensorboard_logger, **kwargs):
        self.args = args
        self.datas = datas
        self.log_dir = log_dir
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        netE_kwargs,netL_kwargs,netD_kwargs = self.build_models(args)
        self.counters = {'t':1, 'epoch':1, 'now_t':1}
        self.iter_per_epoch = datas['iter_per_epoch']
        self.checkpoint = self.initialize_checkpoint(args, netE_kwargs,netL_kwargs,netD_kwargs)
        if self.args.load_pretrained in ['continue', 'pretrained']:
            self.restore_models()
        self.running_paras = {'output_attentions':self.args.output_attentions, 'output_all_states':self.args.output_all_states, 'keep_multihead_output':self.args.keep_multihead_output}
        tmp_dir = os.path.join('./tmp', self.args.log_id)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir

    def build_models(self, args):
        self.logger.info('Defining Network Structure...')
        self.grad_clip = self.args.grad_clip_value > 0
        # Encoder
        netE_kwargs = {
            'model_type':args.model_type, 'layers_setting':args.enc_layers_setting, 'embed_layers_setting':args.embed_layers_setting, 'input_dim': args.input_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length, 'max_size':args.max_size, 'type_size':args.type_size,
            'conditional':args.conditional, 'position_type':args.position_type, 'segment_type':args.segment_type, 'sketch_embed_type':args.sketch_embed_type, 'embed_pool_type':args.embed_pool_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        netE = SketchVAEEncoder(**netE_kwargs)
        # Latent Embedding
        netL_kwargs = {
            'hidden_dim':args.enc_layers_setting[0][1],
            'latent_dim':args.latent_dim,
            'max_length':args.max_length
        }
        netL = SketchVAELatentEmbedding(**netL_kwargs)

        if 'maskrec' in args.task_types:
            output_dim = args.input_dim
        elif 'maskgmm' in args.task_types:
            output_dim = 6*args.M + 3

        # Decoder
        netD_kwargs = {
            'model_type':args.model_type, 'layers_setting':args.dec_layers_setting, 'embed_layers_setting':args.embed_layers_setting,'rec_layers_setting':args.rec_layers_setting, 'input_dim': args.input_dim, 'output_dim':output_dim, 'latent_dim':args.latent_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length, 'max_size':args.max_size, 'type_size':args.type_size,
            'conditional':args.conditional, 'position_type':args.position_type, 'segment_type':args.segment_type, 'sketch_embed_type':args.sketch_embed_type, 'embed_pool_type':args.embed_pool_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        netD = SketchVAEDecoder(**netD_kwargs)

        self.logger.info("Sketch Transformer Sturture:\n{}\n{}".format(netE, netD))

        self.models = {'netE':netE, 'netL':netL, 'netD':netD}

        self.gpu_ids = {'netE':torch.device('cuda:{}'.format(args.gpu_ids[0])),
                        'netL':torch.device('cuda:{}'.format(args.gpu_ids[0])),
                        'netD':torch.device('cuda:{}'.format(args.gpu_ids[0])),}

        # Change to gpu
        for key in self.models:
            self.models[key] = self.models[key].to(self.gpu_ids[key])

        self.logger.info('Finish Construction of Networks.')

        # Defining Optimizers
        self.opts = {
            'optE': torch.optim.Adam(netE.parameters(), lr=args.learning_rate),
            'optL': torch.optim.Adam(netL.parameters(), lr=args.learning_rate),
            'optD': torch.optim.Adam(netD.parameters(), lr=args.learning_rate),
        }


        self.logger.info('Finish Construction of Optimizers.')

        # Defining losses
        self.logger.info('Defining Losses...')
        self.losses = {'mask_l1':F.l1_loss, 'mask_type':nn.CrossEntropyLoss(), 'mask_gmm':GMMLoss(), 'kl':KLLoss(args.kl_tolerance)}

        self.losses_weights = {'mask_l1':args.mask_l1_weight, 'mask_type':args.mask_type_weight,  'mask_gmm':args.mask_gmm_weight, 'kl':args.kl_weight_start}
        self.valid_losses = [key for key in self.losses_weights if self.losses_weights[key] > 0]
        self.logger.info('Finish Losses.')

        return netE_kwargs, netL_kwargs, netD_kwargs

    def initialize_checkpoint(self, args, netE_kwargs, netL_kwargs, netD_kwargs):
        checkpoint = {
            'args': args.__dict__,
            'netE_kwargs': netE_kwargs,
            'netL_kwargs': netL_kwargs,
            'netD_kwargs': netD_kwargs,
            'counters': {
                't': None,
                'epoch': None,
            },
            'netE_state': None, 'netE_best_state': None, 'optE_state': None,
            'netL_state': None, 'netL_best_state': None, 'optL_state': None,
            'netD_state': None, 'netD_best_state': None, 'optD_state': None,
            'best_t': [],
        }
        return checkpoint

    def initialize_validate_losses(self):
        total_losses = {}
        if 'maskrec' in self.args.task_types:
            types = ['mask_l1', 'rec_l1', 'mask_type', 'rec_type']
            for t in types:
                if t in self.valid_losses:
                    total_losses[t] = AverageMeter()
        if 'maskgmm' in self.args.task_types:
            types = ['mask_gmm', 'rec_gmm', 'mask_type', 'rec_type']
            for t in types:
                if t in self.valid_losses:
                    total_losses[t] = AverageMeter()

        if 'maskdisc' in self.args.task_types:
            types = ['x_mask_disc', 'y_mask_disc', 'type_mask_disc', 'x_rec_disc', 'y_rec_disc', 'type_rec_disc']
            for t in types:
                if t in self.valid_losses:
                    total_losses[t] = AverageMeter()
        return total_losses

    def add_decode_prefix(self, input_states):
        decode_prefix = torch.zeros(1, input_states.size(1), input_states.size(2))
        decode_prefix[:,:,2] = 1
        decode_input_states = torch.cat([decode_prefix.to(input_states.device), input_states[:input_states.size(0),:,:]], dim=0)
        return decode_input_states

    def generate_attention_mask(self, length_mask, mask_type):
        batch_size, max_len = length_mask.size()
        attention_mask = length_to_mask(torch.arange(1,max_len+1), max_len=max_len, dtype=torch.long).view(1,max_len,max_len).repeat(batch_size,1,1).to(length_mask.device)
        #print(attention_mask[0])
        return attention_mask

    def update_kl_weight(self, kl_weight, kl_weight_start, kl_decay_rate, step):
        curr_kl_weight = kl_weight - (kl_weight-kl_weight_start)*(kl_decay_rate)**step
        return curr_kl_weight
    def lazy_update_kl_weight(self, curr_kl_weight, kl_weight, kl_decay_rate):
        curr_kl_weight = kl_weight - (kl_weight-curr_kl_weight)*kl_decay_rate
        return curr_kl_weight
    '''
        Training: only for training process.
        input:
            None: Take From Trainer Itself.
    '''
    def train(self):
        netE, netL, netD = self.models['netE'],self.models['netL'],self.models['netD']
        optE, optL, optD = self.opts['optE'],self.opts['optL'],self.opts['optD']

        for epoch_i in range(self.args.num_epoch):
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['train_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time
                self.counters['now_t'] = data_i + 1

                # Tranining Process
                batch_data = [tensor.to(device=self.gpu_ids['netE'],dtype=torch.float32) for tensor in batch_data]
                mask_input_state, input_state, segment, length_mask, input_mask, target = batch_data

                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segment = segment.to(dtype=torch.long)
                target = target.to(dtype=torch.long)

                enc_state = netE(input_state, length_mask, targets=target, segments=segment, head_mask=None, **self.running_paras)

                if self.args.output_attentions:
                    enc_state, attention_prob = enc_state

                mu, sigma, z = netL(enc_state, length_mask)

                # Generate attention mask

                attention_mask = self.generate_attention_mask(length_mask, self.args.train_mode)

                # rec_states[batch, seq_len, 6]
                rec_state = netD(input_state, z, attention_mask, targets=target, head_mask=None, **self.running_paras)

                # Size Checking: rec_states[batch, seq_len, hidden_dim],
                total_loss, losses = self.calculate_losses(input_state, length_mask, rec_state, mu, sigma)

                # Optimizing
                optE.zero_grad(), optL.zero_grad(), optD.zero_grad()
                total_loss.backward()
                if self.grad_clip:
                    clip_grad_value_(netE.parameters(), self.args.grad_clip_value), clip_grad_value_(netL.parameters(), self.args.grad_clip_value), clip_grad_value_(netD.parameters(), self.args.grad_clip_value)
                optE.step(), optL.step(), optD.step()

                # Update results
                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

                if self.counters['t'] % self.args.print_every == 0:
                    evaluations = self.evaluate()
                    self.update_log('train', losses, evaluations)

                if self.counters['t'] % self.args.checkpoint_every == 0:
                    #self.update_img_results('train', imgs, imgs_pred[-1], segs_image)
                    self.validate()
                    self.update_checkpoint()
                    self.save_checkpoint('latest')

                if self.counters['t'] % self.args.save_model_every == 0:
                    self.save_checkpoint('iter_{}'.format(self.counters['t']))
                self.losses_weights['kl'] = self.lazy_update_kl_weight(self.losses_weights['kl'], self.args.kl_weight, self.args.kl_decay_rate)
                # self.losses_weights['kl'] = self.update_kl_weight(self.args.kl_weight, self.args.kl_weight_start, self.args.kl_decay_rate, self.counters['t'])
                self.counters['t'] += 1
                start_time = time.time()
            self.counters['epoch'] += 1

    def validate(self):
        netE, netL, netD = self.models['netE'], self.models['netL'], self.models['netD']
        total_losses = self.initialize_validate_losses()

        topk = (1,5)
        with torch.no_grad():
            for data_i, batch_data in enumerate(self.datas['val_loader']):

                # Tranining Process
                batch_data = [tensor.to(device=self.gpu_ids['netE'],dtype=torch.float32) for tensor in batch_data]
                mask_input_state, input_state, segment, length_mask, input_mask, target = batch_data

                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segment = segment.to(dtype=torch.long)
                target = target.to(dtype=torch.long)

                enc_state = netE(input_state, length_mask, targets=target, segments=segment,  head_mask=None, **self.running_paras)

                if self.args.output_attentions:
                    enc_state, attention_prob = enc_state

                mu, sigma, z = netL(enc_state, length_mask)

                # Generate attention mask
                attention_mask = self.generate_attention_mask(length_mask, self.args.train_mode)
                # rec_states[batch, seq_len, 6]
                rec_state = netD(input_state, z, attention_mask, targets=target, head_mask=None, **self.running_paras)

                # Size Checking: rec_states[batch, seq_len, hidden_dim],
                total_loss, losses = self.calculate_losses(input_state, length_mask, rec_state, mu, sigma)

                for key in total_losses:
                    total_losses[key].update(losses[key].item(), input_state.size(0))

        log_losses = {key:torch.tensor(total_losses[key].avg) for key in total_losses}
        log_evaluations = {'No':1}
        # Update Logs
        self.update_log('val', log_losses, log_evaluations)
        stroke_pred = self.add_decode_prefix(rec_state)
        self.update_stroke_results('val', input_state, stroke_pred, input_mask)

    def decode_parameters(self, y, mask):
        # y [batch, seq_len, 6*M+3]
        # mask [batch, seq_len]
        M = int((y.size(2)-3)/6)
        g_paras = y[:, :, :6*M].view(y.size(0), y.size(1), M, 6)
        pis = g_paras[:,:,:, 0]
        mus = g_paras[:,:,:, 1:3]
        sigmas = torch.exp(g_paras[:,:,:, 3:5])
        rhos = torch.tanh(g_paras[:,:,:, 5])
        qs = y[:,:, 6*M:6*M+3]
        if mask is not None:
            return pis[mask==1,:], mus[mask==1,:,:], sigmas[mask==1,:,:], rhos[mask==1,:], qs
        return pis, mus, sigmas, rhos, qs

    def sample_single_sketch(self, max_len):
        '''
        z[latent_dim]
        '''
        gaussian_generator = MultivariateNormal(torch.zeros(self.args.latent_dim), torch.eye(self.args.latent_dim))
        z = gaussian_generator.sample().to(self.gpu_ids['netD'])
        input_state = torch.zeros(1, max_len, 5).to(self.gpu_ids['netD'])
        input_state[:,0,2] = 1
        attention_mask = self.generate_attention_mask(torch.ones(1,max_len), 'generation').to(self.gpu_ids['netD'])
        netD = self.models['netD']
        seq_len = 1
        for i in range(max_len-1):
            rec_state = netD(input_state, z.view(1,-1), attention_mask, head_mask=None, **self.running_paras)
            pred_state = rec_state[0,i]
            if 'maskrec' in self.args.task_types:
                pred_state[2:5] = (pred_state[2:5] == pred_state.max(dim=0, keepdim=True)[0])
                if pred_state[4] == 1:
                    break
                input_state[:,i+1] = pred_state
            elif 'maskgmm' in self.args.task_types:
                pis, mus, sigmas, rhos, qs = self.decode_parameters(pred_state.view(1,1,-1), None)
                #print(pis.size(), mus.size(), sigmas.size(), rhos.size(), qs.size())
                input_state[:,i+1] = self.sample_single_stroke(pis.squeeze(), mus.squeeze(), sigmas.squeeze(), rhos.squeeze(), qs.squeeze(), self.args.gamma)
            seq_len = i+2
        return input_state[0].detach().cpu().numpy(), seq_len

    def sample_single_stroke(self, pis, mus, sigmas, rhos, qs, gamma):
        """
        Input:
            pis[M]
            mus[M, 2]
            sigmas[M, 2]
            rhos[M]
            qs[3]
        Output:
            strokes[5]
        """
        comp_m = OneHotCategorical(logits=pis)
        comp_choice = (comp_m.sample()==1)

        mu, sigma, rho, q = mus[comp_choice].view(-1), sigmas[comp_choice].view(-1), rhos[comp_choice].view(-1), qs.view(-1)

        cov = (torch.diag((sigma*sigma)) + (1-torch.eye(2).to(mu.device)) * rho * torch.prod(sigma)).to(device=mu.device)

        normal_m = MultivariateNormal(mu, cov)
        stroke_move = normal_m.sample().to(pis.device) # [seq_len, 2]
        pen_states = (q == q.max(dim=0, keepdim=True)[0]).to(dtype=torch.float)#[seq_len, 3]
        # print('mu',mu,'stroke_move', stroke_move, 'pen_states', pen_states)
        stroke = torch.cat([stroke_move.view(-1), pen_states.view(-1)], dim=0).to(pis.device)
        return stroke

    def recon_sketches(self, pis, mus, sigmas, rhos, qs, gamma):
        """
        Input:
            pis[batch, seq_len, M]
            mus[batch, seq_len, M, 2]
            sigmas[batch, seq_len, M, 2]
            rhos[batch, seq_len, M]
            qs[batch, seq_len, 3]
        Output:
            strokes[batch, seq_len, 5]:
        """
        batch_size, seq_len, M = pis.size()
        sketches = []
        sigmas = sigmas * gamma
        # Sample for each sketch
        for i in range(batch_size):
            strokes = []
            #print(pis[:,i,:].size(), pis[:,i,:].device)
            #print(pis.size(), mus.size(), sigmas.size(), rhos.size(), qs.size())
            for j in range(seq_len):
                comp_m = OneHotCategorical(logits=pis[i,j])
                comp_choice = (comp_m.sample()==1)

                mu, sigma, rho, q = mus[i,j][comp_choice].view(-1), sigmas[i,j][comp_choice].view(-1), rhos[i,j][comp_choice].view(-1), qs[i,j].view(-1)

                cov = (torch.diag((sigma*sigma)) + (1-torch.eye(2).to(mu.device)) * rho * torch.prod(sigma)).to(device=mu.device)

                normal_m = MultivariateNormal(mu, cov)
                stroke_move = normal_m.sample().to(pis.device) # [seq_len, 2]
                pen_states = (q == q.max(dim=0, keepdim=True)[0]).to(dtype=torch.float)#[seq_len, 3]

                stroke = torch.cat([stroke_move.view(-1), pen_states.view(-1)], dim=0).to(pis.device)
                strokes.append(stroke)
            sketches.append(torch.stack(strokes))
        return torch.stack(sketches)

    def calculate_losses(self, input_state, length_mask, rec_state, mu_latent, sigma_latent):
        # Intialize total_loss and losses
        losses = {}
        total_loss = torch.zeros(1).to(self.gpu_ids['netE'])
        if 'maskgmm' in self.args.task_types:
            #print(rec_state.size())
            pis, mus, sigmas, rhos, qs = self.decode_parameters(rec_state[:,:-1,:], None)
            losses['mask_gmm'] = self.losses_weights['mask_gmm'] * self.losses['mask_gmm'](input_state[:,1:,:2], length_mask[:,1:], pis, mus, sigmas, rhos)

            total_loss = total_loss + losses['mask_gmm']

            if 'mask_type' in self.valid_losses:
                losses['mask_type'] = self.losses_weights['mask_type'] * self.losses['mask_type'](qs.contiguous().view(-1,3), torch.argmax(input_state[:, 1:, 2:5], dim=2).view(-1))
                total_loss = total_loss + losses['mask_type']

        if 'maskrec' in self.args.task_types:
            losses['mask_l1'] = self.losses_weights['mask_l1'] * self.losses['mask_l1'](rec_state[:, :-1, 0:2][length_mask[:,1:]==1,:], input_state[:, 1:, 0:2][length_mask[:,1:]==1,:])

            total_loss = total_loss + losses['mask_l1']

            if 'mask_type' in self.valid_losses:
                losses['mask_type'] = self.losses_weights['mask_type'] * self.losses['mask_type'](rec_state[:, :-1, 2:5].contiguous().view(-1,3), torch.argmax(input_state[:, 1:, 2:5], dim=2).view(-1))
                total_loss = total_loss + losses['mask_type']

        if 'kl' in self.valid_losses:
            losses['kl'] = self.losses_weights['kl'] * self.losses['kl'](mu_latent, sigma_latent)
            total_loss = total_loss +  losses['kl']


        return total_loss, losses

    def evaluate(self, topk=(1,5)):
        results = {'No':1}
        return results

    def restore_models(self):
        netE, netL, netD = self.models['netE'], self.models['netL'], self.models['netD']
        optE, optL, optD = self.opts['optE'], self.opts['optL'], self.opts['optD']
        restore_path = self.args.restore_checkpoint_path
        self.logger.info('Restoring from checkpoint:{}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        if 'enc_net_from_mask' in self.args.which_pretrained:
            netE.load_model(checkpoint['netE_state'], True)
        if 'enc_net' in self.args.which_pretrained:
            netE.load_model(checkpoint['netE_state'], False)
        if 'enc_opt' in self.args.which_pretrained:
            optE.load_state_dict(checkpoint['optE_state'])

        if 'latent_net' in self.args.which_pretrained:
            netL.load_state_dict(checkpoint['netL_state'])
        if 'latent_opt' in self.args.which_pretrained:
            optL.load_state_dict(checkpoint['optL_state'])

        if 'dec_net_from_mask' in self.args.which_pretrained:
            netD.load_model(checkpoint['netE_state'], True)
        if 'dec_net' in self.args.which_pretrained:
            netD.load_model(checkpoint['netD_state'], False)
        if 'dec_opt' in self.args.which_pretrained:
            optD.load_state_dict(checkpoint['optD_state'])

        if 'continue' == self.args.load_pretrained:
            self.checkpoint['counters']['t'] = checkpoint['counters']['t']
            self.checkpoint['counters']['epoch'] = checkpoint['counters']['epoch']



    def save_checkpoint(self, name):
        checkpoint_path = '{}/{}_ckpt.pth.tar'.format(self.log_dir, name)
        self.logger.info('Saving checkpoint to {}'.format(checkpoint_path) )
        torch.save(self.checkpoint, checkpoint_path)

    def update_checkpoint(self):
        self.checkpoint['netE_state'] = self.models['netE'].state_dict()
        self.checkpoint['optE_state'] = self.opts['optE'].state_dict()
        self.checkpoint['netL_state'] = self.models['netL'].state_dict()
        self.checkpoint['optL_state'] = self.opts['optL'].state_dict()
        self.checkpoint['netD_state'] = self.models['netD'].state_dict()
        self.checkpoint['optD_state'] = self.opts['optD'].state_dict()
        self.checkpoint['counters']['t'] = self.counters['t']
        self.checkpoint['counters']['epoch'] = self.counters['epoch']
    '''
        Showing results on terminal
    '''
    def display_results(self, mode, losses, evaluations):
        if mode == 'train':
            result_message = 'Train Epoch {}, Time:{:.3f}[T:{:.3f}/D:{:.3f}], Total Iter [{}/{}], [{}/{}]:'.format(self.counters['epoch'], self.batch_time, self.run_time, self.data_time, self.counters['t'], self.args.num_iterations, self.counters['now_t'], self.iter_per_epoch)
        elif mode == 'val':
            result_message = 'Validate Epoch {}, Time:{:.3f}[T:{:.3f}/D:{:.3f}], Total Iter [{}/{}], [{}/{}]:'.format(self.counters['epoch'], self.batch_time, self.run_time, self.data_time, self.counters['t'], self.args.num_iterations, self.counters['now_t'], self.iter_per_epoch)
        loss_message = ', '.join(['[{}_{}]:{:.4f}'.format(mode, key, loss.item()) for key, loss in losses.items()])
        evaluations_message = ', '.join(['[{}_{}]:{:.4f}'.format(mode, key, evaluation) for key, evaluation in evaluations.items()])
        message = result_message + 'Losses:' + loss_message + ', Evaluations:' + evaluations_message
        return message

    '''
        Update Log and update tensorboard
    '''
    def update_log(self, mode, losses, evaluations):
        # Display first
        self.logger.info(self.display_results(mode, losses, evaluations))

        # Update Losses log
        if losses is not None:
            for key, loss in losses.items():
                self.tensorboard_logger.add_scalar('{}_{}'.format(mode, key), loss.item(), self.counters['t'])
        if evaluations is not None:
            for key, evaluation in evaluations.items():
                self.tensorboard_logger.add_scalar('{}_{}'.format(mode, key), evaluation, self.counters['t'])

    def is_valid_sketch(self, sketch):
        return np.all(sketch[:,:2].sum(axis=0) != 0)
    '''
        Update Log and update tensorboard image results
    '''
    def update_stroke_results(self, mode, strokes, strokes_recon, masks, size=128):

        imgs = []
        imgs_pred = []
        imgs_recon = []
        num_display_samples = min(len(strokes), self.args.num_display_samples)
        if 'maskgmm' in self.args.task_types:
            pis, mus, sigmas, rhos, qs = self.decode_parameters(strokes_recon, None)
            strokes_recon = self.recon_sketches(pis, mus, sigmas, rhos, qs, self.args.gamma)
        else:
            strokes_recon_type = strokes_recon[:,:,2:5]
            strokes_recon[:,:,2:5] = strokes_recon_type == strokes_recon_type.max(dim=2, keepdim=True)[0]

        for i in range(num_display_samples):
            tmp_stroke = strokes[i].cpu().numpy()
            tmp_stroke_recon = strokes_recon[i].cpu().numpy()
            tmp_stroke_pred, pred_seq_len = self.sample_single_sketch(self.args.max_length)
            if self.args.stroke_type == 'stroke-5':
                tmp_stroke = to_normal_strokes(tmp_stroke)
                tmp_stroke_recon = to_normal_strokes(tmp_stroke_recon)
                tmp_stroke_pred = to_normal_strokes(tmp_stroke_pred)
            if pred_seq_len == 1 or len(tmp_stroke_recon) == 1 or not self.is_valid_sketch(tmp_stroke_pred) or not self.is_valid_sketch(tmp_stroke_recon):
                continue
            imgs.append(strokes2drawing(tmp_stroke,  size=128, svg_filename='{}/tmp_{}.svg'.format(self.tmp_dir, i)))
            imgs_recon.append(strokes2drawing(tmp_stroke_recon,  size=128, svg_filename='{}/tmp_recon_{}.svg'.format(self.tmp_dir, i)))
            imgs_pred.append(strokes2drawing(tmp_stroke_pred,  size=128, svg_filename='{}/tmp_pred_{}.svg'.format(self.tmp_dir, i)))
        if len(imgs) == 0:
            return
        imgs, imgs_recon, imgs_pred = np.array(imgs), np.array(imgs_recon), np.array(imgs_pred)
        #print(imgs.shape, imgs_pred.shape)
        record_imgs = np.concatenate([imgs, imgs_recon, imgs_pred], axis=2)

        for i in range(len(record_imgs)):
            #print(i)
            self.tensorboard_logger.add_image('{}_record_imgs_{}'.format(mode,i), record_imgs[i].reshape((1,)+record_imgs[i].shape), self.counters['t'])
