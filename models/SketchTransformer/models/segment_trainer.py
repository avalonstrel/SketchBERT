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
from .networks import SketchSegmentTransformerModel, MaskSketchRecModel, MaskSketchGMMModel, SketchClassificationModel, SketchClsPoolingModel, SketchRetrievalPoolingModel, SketchDiscretePoolingModel, SketchSegmentOrderPoolingModel, GMMLoss
from .utils import segments2sequence, strokes2drawing, to_normal_strokes, rec_incomplete_strokes

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical


TASKMODELS = {'maskrec':MaskSketchRecModel, 'maskgmm':MaskSketchGMMModel, 'maskdisc':SketchDiscretePoolingModel, 'sketchcls':SketchClassificationModel, 'sketchclsinput':SketchClsPoolingModel, 'sketchretrieval':SketchRetrievalPoolingModel,'segorder':SketchSegmentOrderPoolingModel}

class SketchSegmentTransformerTrainer(Trainer):
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
        self.cls_in_input = 'sketchclsinput' in args.task_types
        self.rel_in_input = 'sketchretrieval' in args.task_types
        self.tensorboard_logger = tensorboard_logger
        cls_in_input = 'sketchclsinput' in args.task_types
        netE_kwargs = self.build_models(args)
        self.counters = {'t':1, 'epoch':1, 'now_t':1}
        self.iter_per_epoch = datas['iter_per_epoch']
        self.checkpoint = self.initialize_checkpoint(args, netE_kwargs)
        if self.args.load_pretrained in ['continue', 'pretrained']:
            self.restore_models()
        self.running_paras = {'output_attentions':self.args.output_attentions, 'output_all_states':self.args.output_all_states, 'keep_multihead_output':self.args.keep_multihead_output}
        tmp_dir = os.path.join('./tmp', self.args.log_id)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir

    def build_models(self, args):

        self.logger.info('Defining Network Structure...')
        # Encoder
        netE_kwargs = {
            'layers_setting':args.layers_setting, 'input_dim': args.input_dim, 'max_length':args.max_length+self.cls_in_input+self.rel_in_input, 'max_segment':args.max_segment+2,
            'segment_atten_type':args.segment_atten_type, 'position_type':args.position_type, 'segment_type':args.segment_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        netE = SketchSegmentTransformerModel(**netE_kwargs)

        self.logger.info("Sketch Transformer Sturture:\n{}".format(netE))

        self.models = {'netE':netE, }

        self.gpu_ids = {'netE':torch.device('cuda:{}'.format(args.gpu_ids[0])),}


        TASKPARAS = {'maskrec':{'hidden_dim':args.hidden_dim, 'input_dim':args.input_dim, 'cls_in_input':self.cls_in_input, 'rel_in_input':self.rel_in_input},
                     'maskgmm':{'hidden_dim':args.hidden_dim, 'M':args.M, 'cls_in_input':self.cls_in_input, 'rel_in_input':self.rel_in_input},
                     'maskdisc':{'hidden_dim':args.hidden_dim, 'max_size':args.max_size, 'type_size':args.type_size, 'cls_in_input':self.cls_in_input, 'rel_in_input':self.rel_in_input},
                     'sketchcls':{'hidden_dim':args.hidden_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length+self.cls_in_input},
                     'sketchclsinput':{'hidden_dim':args.hidden_dim, 'cls_dim':args.cls_dim, 'pool_dim':self.rel_in_input},
                     'sketchretrieval':{'hidden_dim':args.hidden_dim, 'feat_dim':args.rel_feat_dim, 'pool_dim':0},
                     'segorder':{'hidden_dim':args.hidden_dim, 'max_segment':args.max_segment, 'cls_in_input':self.cls_in_input, 'rel_in_input':self.rel_in_input}}

        for task in args.task_types:
            self.models[task] = TASKMODELS[task](**TASKPARAS[task])
            self.gpu_ids[task] = torch.device('cuda:{}'.format(args.gpu_ids[0]))

        # Change to gpu
        for key in self.models:
            self.models[key] = self.models[key].to(self.gpu_ids[key])

        self.logger.info('Finish Construction of Networks.')

        # Defining Optimizers
        self.opts = {
            'optE': torch.optim.Adam(netE.parameters(), lr=args.learning_rate),
        }

        for task in args.task_types:
            self.opts[task] = torch.optim.Adam(self.models[task].parameters(), lr=args.learning_rate)

        self.logger.info('Finish Construction of Optimizers.')

        # Defining losses
        self.logger.info('Defining Losses...')
        self.losses = {'mask_axis':F.l1_loss, 'mask_disc':nn.CrossEntropyLoss(),
                       'mask_type':nn.CrossEntropyLoss(), 'prediction':nn.CrossEntropyLoss(), 'gmm':GMMLoss(),
                       'triplet':nn.TripletMarginLoss(margin=args.triplet_margin), }

        self.losses_weights = {'mask_axis':args.mask_axis_weight, 'rec_axis':args.rec_axis_weight,
                               'mask_type':args.mask_type_weight, 'rec_type':args.rec_type_weight,
                               'mask_gmm':args.mask_gmm_weight,  'rec_gmm':args.rec_gmm_weight,
                               'prediction':args.prediction_weight, 'triplet':args.triplet_weight, 'segorder':args.segorder_weight}
        self.logger.info('Finish Losses.')

        return netE_kwargs

    def initialize_checkpoint(self, args, netE_kwargs):
        checkpoint = {
            'args': args.__dict__,
            'netE_kwargs': netE_kwargs,
            'counters': {
                't': None,
                'epoch': None,
            },
            'netE_state': None, 'netE_best_state': None, 'optE_state': None,
            'best_t': [],
        }
        for task in args.task_types:
            checkpoint['net_{}_state'.format(task)] = None
            checkpoint['net_{}_best_state'.format(task)] = None
            checkpoint['opt_{}_state'.format(task)] = None
        return checkpoint

    def restore_models(self):
        netE = self.models['netE']
        optE = self.opts['optE']
        restore_path = self.args.restore_checkpoint_path
        self.logger.info('Restoring from checkpoint:{}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        if 'enc_net' in self.args.which_pretrained:

            netE.load_model(checkpoint['netE_state'], self.rel_in_input, self.cls_in_input, 'sketchretrieval' in checkpoint['args']['_cfg_dict']['task_types'], 'sketchclsinput' in checkpoint['args']['_cfg_dict']['task_types'])
            self.logger.info('Restoring netE')
        if 'task_net' in self.args.which_pretrained:
            for task in self.args.task_types:
                if 'net_{}_state'.format(task) in checkpoint:
                    self.models[task].load_state_dict(checkpoint['net_{}_state'.format(task)])
            self.logger.info('Restoring net tasks')
        if 'enc_opt' in self.args.which_pretrained:
            optE.load_state_dict(checkpoint['optE_state'])
        if 'task_opt' in self.args.which_pretrained:
            for task in self.args.task_types:
                if 'opt_{}_state'.format(task) in checkpoint:
                    self.opts[task].load_state_dict(checkpoint['opt_{}_state'.format(task)])

        if 'continue' == self.args.load_pretrained:
            self.checkpoint['counters']['t'] = checkpoint['counters']['t']
            self.checkpoint['counters']['epoch'] = checkpoint['counters']['epoch']

    def remove_segment(self, stroke_segment, segment_index):
        length = (segment_index==1).sum(dim=1)
        length_mask = length_to_mask(length, max_len=segment_index.size(1), dtype=torch.long)
        removed_stroke_segment = torch.zeros(*stroke_segment.size()).to(device=stroke_segment.device)
        removed_stroke_segment[length_mask == 1, :] = stroke_segment[segment_index == 1, :]
        outside_state = removed_stroke_segment[length_mask == 0]
        outside_state[:,4] = 1
        removed_stroke_segment[length_mask==0] = outside_state
        return removed_stroke_segment
    '''
        Training: only for training process.
        input:
            None: Take From Trainer Itself.
    '''
    def train(self):
        netE = self.models['netE']
        optE = self.opts['optE']

        for epoch_i in range(self.args.num_epoch):
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['train_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time
                self.counters['now_t'] = data_i + 1

                # Tranining Process
                if 'sketchretrieval' not in self.args.task_types:
                    batch_data = [[term ,] for term in batch_data]

                batch_data = [[t.to(device=self.gpu_ids['netE'],dtype=torch.float32) for t in tensor ] for tensor in batch_data]
                masked_stroke_segments, stroke_segments, segments, segment_indexs, segment_atten_masks, input_masks, targets = batch_data
                #print(masked_stroke_segments[0].size())
                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segments = [segment.to(dtype=torch.long) for segment in segments]
                targets = [t.to(dtype=torch.long) for t in targets]

                output_states, pooled_outputs = [], []
                for masked_stroke_segment, segment_atten_mask, segment, segment_index in zip(masked_stroke_segments, segment_atten_masks, segments, segment_indexs):
                    #print(masked_stroke_segment.size())
                    output_state = netE(masked_stroke_segment, segment, segment_atten_mask, segment_index, head_mask=None, **self.running_paras)

                    if self.args.output_attentions:
                        output_state, attention_prob = output_state
                    output_states.append(output_state)

                    pooled_output = {task:self.models[task](output_state) for task in self.args.task_types if task != 'segorder'}
                    if 'segorder' in self.args.task_types:
                        pooled_output['segorder'] = self.models['segorder'](output_state, segment_index)
                    pooled_outputs.append(pooled_output)

                # Size Checking: output_states, pooled_output[[batch, seq_len, 6*M+3],[]]
                total_loss, losses = self.calculate_losses(stroke_segments, segment_atten_masks, segments, segment_indexs, input_masks, pooled_outputs, targets)

                # Optimizing
                optE.zero_grad()
                for task in self.args.task_types:
                    self.opts[task].zero_grad()
                total_loss.backward()
                optE.step()
                for task in self.args.task_types:
                    self.opts[task].step()

                # Update results
                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

                if self.counters['t'] % self.args.print_every == 0:
                    evaluations = None
                    self.evaluate(pooled_outputs, targets)
                    self.update_log('train', losses, evaluations)

                if self.counters['t'] % self.args.checkpoint_every == 0:
                    #self.update_img_results('train', imgs, imgs_pred[-1], segs_image)
                    self.validate()
                    self.update_checkpoint()
                    self.save_checkpoint('latest')

                if self.counters['t'] % self.args.save_model_every == 0:
                    self.save_checkpoint('iter_{}'.format(self.counters['t']))
                self.counters['t'] += 1
                start_time = time.time()
            self.counters['epoch'] += 1

    def validate(self):
        netE = self.models['netE']
        total_losses = self.initialize_validate_losses()
        topk = (1,5)
        total_evaluations = {'accuracy_{}'.format(k): AverageMeter() for k in topk}
        total_strokes, total_strokes_pred, total_input_masks, total_retrieval_outputs, total_targets = [], [], [], [], []
        with torch.no_grad():
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['val_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time
                if 'sketchretrieval' not in self.args.task_types:
                    batch_data = [[term ,] for term in batch_data]

                batch_data = [[t.to(device=self.gpu_ids['netE'],dtype=torch.float32) for t in tensor ] for tensor in batch_data]
                masked_stroke_segments, stroke_segments, segments, segment_indexs, segment_atten_masks, input_masks, targets = batch_data

                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segments = [segment.to(dtype=torch.long) for segment in segments]
                targets = [t.to(dtype=torch.long) for t in targets]

                output_states, pooled_outputs = [], []
                for masked_stroke_segment, segment_atten_mask, segment, segment_index in zip(masked_stroke_segments, segment_atten_masks, segments, segment_indexs):
                    output_state = netE(masked_stroke_segment, segment, segment_atten_mask, segment_index, head_mask=None, **self.running_paras)

                    if self.args.output_attentions:
                        output_state, attention_prob = output_state
                    output_states.append(output_state)

                    pooled_output = {task:self.models[task](output_state) for task in self.args.task_types if task != 'segorder'}
                    if 'segorder' in self.args.task_types:
                        pooled_output['segorder'] = self.models['segorder'](output_state, segment_index)
                    pooled_outputs.append(pooled_output)

                # Size Checking: output_states, pooled_output[[batch, seq_len, 6*M+3],[]]
                # All for display, so all need remove segment
                strokes_pred = None
                if 'maskrec' in self.args.task_types:
                    strokes_pred = self.remove_segment(pooled_outputs[0]['maskrec'], segment_indexs[0]).cpu()
                if 'maskgmm' in self.args.task_types:
                    strokes_pred = self.remove_segment(pooled_outputs[0]['maskgmm'], segment_indexs[0]).cpu()
                if 'maskdisc' in self.args.task_types:
                    strokes_pred = self.remove_segment(torch.cat([pooled_outputs[0]['maskdisc'][0].argmax(dim=2,keepdim=True), pooled_outputs[0]['maskdisc'][1].argmax(dim=2,keepdim=True), pooled_outputs[0]['maskdisc'][2].argmax(dim=2,keepdim=True)], dim=2).to(dtype=torch.float), segment_indexs[0]).cpu()

                total_strokes.append(self.remove_segment(stroke_segments[0], segment_indexs[0]).cpu())
                total_strokes_pred.append(strokes_pred)
                total_input_masks.append(self.remove_segment(input_masks[0], segment_indexs[0]).cpu())

                total_targets.append(targets[0])
                total_loss, losses = self.calculate_losses(stroke_segments, segment_atten_masks, segments, segment_indexs, input_masks, pooled_outputs, targets)
                acc_evaluations = None
                if 'sketchcls' in self.args.task_types:
                    acc_evaluations = self.accuracy_evaluation(pooled_outputs[0]['sketchcls'], targets[0])
                if 'sketchclsinput' in self.args.task_types:
                    acc_evaluations = self.accuracy_evaluation(pooled_outputs[0]['sketchclsinput'], targets[0])
                if 'sketchretrieval' in self.args.task_types:
                    total_retrieval_outputs.append(pooled_outputs[0]['sketchretrieval'])
                if losses is not None:
                    for key in total_losses:
                        total_losses[key].update(losses[key].item(), stroke_segments[0].size(0))
                if acc_evaluations is not None:
                    for key in total_evaluations:
                        total_evaluations[key].update(acc_evaluations[key], stroke_segments[0].size(0))
                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

        retrieval_evaluations = {}
        if 'sketchretrieval' in self.args.task_types:
            retrieval_evaluations = self.retrieval_evaluation(torch.cat(total_retrieval_outputs, dim=0), torch.cat(total_targets, dim=0), topk=topk)
        log_losses = {key:torch.tensor(total_losses[key].avg) for key in total_losses}
        log_evaluations = {key:total_evaluations[key].avg for key in total_evaluations}
        log_evaluations = {**log_evaluations, **retrieval_evaluations}
        # Update Logs
        self.update_log('val', log_losses, log_evaluations)

        if total_strokes_pred[0] is not None:
            total_strokes = torch.cat(total_strokes, dim=0)
            total_strokes_pred = torch.cat(total_strokes_pred, dim=0)
            total_input_masks = torch.cat(total_input_masks, dim=0)
            self.update_stroke_results('val', total_strokes, total_strokes_pred, total_input_masks)



    def calculate_losses(self, stroke_segments, segment_atten_masks, segments, segment_indexs, input_masks, pooled_outputs, targets):
        # Intialize total_loss and losses
        losses = {}
        total_loss = torch.zeros(1).to(self.gpu_ids['netE'])

        stroke_segment, segment_atten_mask, segment, segment_index, input_mask, pooled_output, target = stroke_segments[0], segment_atten_masks[0], segments[0], segment_indexs[0], input_masks[0], pooled_outputs[0], targets[0]

        length = (segment_index==1).sum(dim=1)
        length_mask = length_to_mask(length, max_len=segment_index.size(1), dtype=torch.long)
        # Input Stroke
        input_state = self.remove_segment(stroke_segment, segment_index)
        if 'maskgmm' in self.args.task_types:
            y = pooled_output['maskgmm']
            y[length_mask==1,:] = y[segment_index==1,:] # remove the segment feat
            pis, mus, sigmas, rhos, qs = self.decode_parameters(y)
            losses['mask_gmm'] = self.losses_weights['mask_gmm'] * self.losses['gmm'](input_state[:,:,:2], length_mask, pis, mus, sigmas, rhos, input_mask[:,:,0]) / (1-self.args.mask_prob)
            losses['rec_gmm'] = self.losses_weights['rec_gmm'] * self.losses['gmm'](input_state[:,:,:2], length_mask, pis, mus, sigmas, rhos, 1-input_mask[:,:,0]) / (self.args.mask_prob)
            total_loss = total_loss + losses['mask_gmm'] + losses['rec_gmm']

            losses['mask_type'] = self.losses_weights['mask_type'] * self.losses['mask_type'](qs[input_mask[:, :, 2] == 0,:], torch.argmax(input_state[:, :, 2:5], dim=2)[input_mask[:, :, 2] == 0]) / (1-self.args.mask_prob)
            losses['rec_type'] = self.losses_weights['rec_type'] * self.losses['mask_type'](qs[input_mask[:, :, 2] == 1,:], torch.argmax(input_state[:, :, 2:5], dim=2)[input_mask[:, :, 2] == 1]) / (1-self.args.mask_prob)
            total_loss = total_loss + losses['mask_type']

        if 'maskrec' in self.args.task_types:
            rec_seg_state = pooled_output['maskrec'][segment_index != 0,:]
            input_seg_state = input_state[segment_index != 0, :]
            input_seg_mask = input_mask[segment_index != 0, :]
            # May be just compute the mask

            losses['mask_axis'] = self.losses_weights['mask_axis'] * self.losses['mask_axis'](rec_seg_state[:,:2][input_seg_mask[:,:2] ==0], input_seg_state[:,:2][input_seg_mask[:,:2]==0])
            losses['rec_axis'] = self.losses_weights['rec_axis'] * self.losses['mask_axis'](rec_seg_state[:,:2][input_seg_mask[:,:2] == 1], input_seg_state[:,:2][input_seg_mask[:,:2]==1])

            total_loss = total_loss + losses['mask_axis'] + losses['rec_axis']

            losses['mask_type'] = self.losses_weights['mask_type'] * self.losses['mask_type'](rec_seg_state[:,2:5][input_seg_mask[:,2] == 0,:], torch.argmax(input_seg_state[:,2:5],dim=1)[input_seg_mask[:,2]==0])
            losses['rec_type'] = self.losses_weights['rec_type'] * self.losses['mask_type'](rec_seg_state[:,2:5][input_seg_mask[:,2] == 1,:], torch.argmax(input_seg_state[:,2:5],dim=1)[input_seg_mask[:,2]==1])

            total_loss = total_loss + losses['mask_type'] + losses['rec_type']

        if 'maskdisc' in self.args.task_types:
            rec_state = []
            for term in pooled_output['maskdisc']:
                new_term = term.clone()
                new_term[length_mask == 1, :] = term[segment_index == 1, :]
                rec_state.append(new_term)
            losses['x_mask_disc'] = self.losses_weights['mask_axis'] * self.losses['mask_disc'](rec_state[0][input_mask[:,:,0] == 0,:], (input_state[:,:, 0][input_mask[:,:,0] == 0]).to(torch.long)) / (1-self.args.mask_prob)
            losses['x_rec_disc'] = self.losses_weights['rec_axis'] * self.losses['mask_disc'](rec_state[0][input_mask[:,:,0] == 1,:], (input_state[:,:, 0][input_mask[:,:,0] == 1]).to(torch.long)) / (self.args.mask_prob)
            losses['y_mask_disc'] = self.losses_weights['mask_axis'] * self.losses['mask_disc'](rec_state[1][input_mask[:,:,1] == 0,:], (input_state[:,:, 1][input_mask[:,:,1] == 0]).to(torch.long)) / (1-self.args.mask_prob)
            losses['y_rec_disc'] = self.losses_weights['rec_axis'] * self.losses['mask_disc'](rec_state[1][input_mask[:,:,1] == 1,:], (input_state[:,:, 1][input_mask[:,:,1] == 1]).to(torch.long)) / (self.args.mask_prob)
            losses['type_mask_disc'] = self.losses_weights['mask_type'] * self.losses['mask_disc'](rec_state[2][input_mask[:,:,2] == 0,:], (input_state[:,:, 2][input_mask[:,:,2] == 0]).to(torch.long)) / (1-self.args.mask_prob)
            losses['type_rec_disc'] = self.losses_weights['rec_type'] * self.losses['mask_disc'](rec_state[2][input_mask[:,:,2] == 1,:], (input_state[:,:, 2][input_mask[:,:,2] == 1]).to(torch.long)) / (self.args.mask_prob)
            total_loss = total_loss + losses['x_mask_disc'] + losses['y_mask_disc'] + losses['type_mask_disc'] + losses['x_rec_disc'] + losses['y_rec_disc'] + losses['type_rec_disc']

        if 'segorder' in self.args.task_types:
            seg_index_state = segment[:,self.cls_in_input+self.rel_in_input:][segment_index==0] - 2
            losses['segorder'] = self.losses_weights['segorder'] * self.losses['prediction'](pooled_output['segorder'], seg_index_state)

        if 'sketchcls' in self.args.task_types:
            losses['prediction'] = self.losses_weights['prediction'] * self.losses['prediction'](pooled_output['sketchcls'], target)
            total_loss = total_loss + losses['prediction']

        if 'sketchclsinput' in self.args.task_types:
            #print(pooled_output['sketchclsinput'].size(), target.size())
            losses['prediction'] = self.losses_weights['prediction'] * self.losses['prediction'](pooled_output['sketchclsinput'], target)
            total_loss = total_loss + losses['prediction']

        if 'sketchretrieval' in self.args.task_types:
            losses['triplet'] = self.losses_weights['triplet'] * self.losses['triplet'](*[pooled_output['sketchretrieval'] for pooled_output in pooled_outputs])
            total_loss = total_loss + losses['triplet']

        return total_loss, losses


    # Test for sketch-to-sketch retrieval
    def sketch_retrieval_evaluation(self, val_loader, topk=(1,5), batch_size=32):
        netE = self.models['netE']
        val_features = []
        targets = []
        with torch.no_grad():
            for data_i, batch_data in enumerate(val_loader):
                batch_data = [tensor.to(device=self.gpu_ids['netE'],dtype=torch.float32) for tensor in batch_data]
                mask_input_states, input_states, segments, length_mask, input_mask, target = [term[0] for term in batch_data]
                target = target.to(dtype=torch.long)

                output_states = netE(input_states, length_mask, segments=segments, head_mask=None, **self.running_paras) #[batch, seq_len, hidden_dim]

                if self.args.output_attentions:
                    output_states, attention_probs = output_states

                retrieval_features = self.models['retrieval'](output_states)
                val_features.append(retrieval_features)
                targets.append(target)
        val_features = torch.cat(val_features, dim=0)
        targets = torch.cat(targets, dim=0)
        return self.retrieval_evaluation(val_features, targets, topk=topk, batch_size=batch_size)

    def retrieval_evaluation(self, test_states, test_targets, collection_states=None, collection_targets=None, topk=(1,5), batch_size=32):
        collection_is_test = False
        if collection_states is None:
            collection_states = test_states
            collection_targets = test_targets
            collection_is_test = True
        num_test_batch = test_states.size(0) // batch_size + 1
        num_collect_batch = collection_states.size(0) // batch_size + 1
        correct = {k:0 for k in  topk}
        for test_i in range(num_test_batch):
            test_i_distances = []
            for collect_i in range(num_collect_batch):
                test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
                collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
                tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
                if collection_is_test and test_i == collect_i:
                    tmp_distances[torch.arange(tmp_distances.size(0)), torch.arange(tmp_distances.size(0))] = 1e+8
                test_i_distances.append(tmp_distances)
            test_i_distances = -torch.cat(test_i_distances, dim=1) #[batch, length]
            _ , pred_indics = test_i_distances.topk(max(topk), dim=1)
            #test_indics = [i for i in range(test_i*batch_size, (test_i+1)*batch_size)]
            #print(test_indics, test_targets)

            test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]
            # pred_i_target = collection_targets[pred_indics]
            for j in range(len(test_i_target)):
                for k in topk:
                    if test_i_target[j] in collection_targets[pred_indics[j][:k]]:
                        correct[k] = correct[k] + 1

        return {'retrieval_{}'.format(k):(correct[k] / test_states.size(0)) for k in topk}

    def sbir_retrieval_evaluation(self):
        pass


    def accuracy_evaluation(self, prediction, target, topk=(1,5)):
        results = {}
        res_acc = accuracy(prediction, target, topk=topk)
        for k in topk:
            results['accuracy_{}'.format(k)] = res_acc[k][0]
        return results

    def evaluate(self, pooled_outputs, targets, topk=(1,5)):
        results = {}
        # Classification evaluation
        res_acc = None
        if 'sketchcls' in self.args.task_types:
            res_acc = accuracy(pooled_outputs[0]['sketchcls'], targets[0], topk=topk)
        if 'sketchclsinput' in self.args.task_types:
            res_acc = accuracy(pooled_outputs[0]['sketchclsinput'], targets[0], topk=topk)
        if res_acc is not None:
            for k in topk:
                results['accuracy_{}'.format(k)] = res_acc[k][0]
        # Retrieval Evaluation
        if 'sketchretrieval' in self.args.task_types:
            retrieval_evaluations = self.retrieval_evaluation(pooled_outputs[0]['sketchretrieval'], targets[0], topk=(1,))
            results = {**results, **retrieval_evaluations}
        return results

    def initialize_validate_losses(self):
        total_losses = {}
        if 'maskrec' in self.args.task_types:
            total_losses['mask_axis'] = AverageMeter()
            total_losses['rec_axis'] = AverageMeter()
            total_losses['mask_type'] = AverageMeter()
            total_losses['rec_type'] = AverageMeter()
        if 'maskgmm' in self.args.task_types:
            total_losses['mask_gmm'] = AverageMeter()
            total_losses['rec_gmm'] = AverageMeter()
            total_losses['mask_type'] = AverageMeter()
            total_losses['rec_type'] = AverageMeter()
        if 'maskdisc' in self.args.task_types:
            total_losses['x_mask_disc'] = AverageMeter()
            total_losses['y_mask_disc'] = AverageMeter()
            total_losses['type_mask_disc'] = AverageMeter()
            total_losses['x_rec_disc'] = AverageMeter()
            total_losses['y_rec_disc'] = AverageMeter()
            total_losses['type_rec_disc'] = AverageMeter()
        if 'sketchcls' in self.args.task_types:
            total_losses['prediction'] = AverageMeter()
        if 'sketchclsinput' in self.args.task_types:
            total_losses['prediction'] = AverageMeter()
        if 'sketchretrieval' in self.args.task_types:
            total_losses['triplet'] = AverageMeter()
        return total_losses

    def save_checkpoint(self, name):
        checkpoint_path = '{}/{}_ckpt.pth.tar'.format(self.log_dir, name)
        self.logger.info('Saving checkpoint to {}'.format(checkpoint_path) )
        torch.save(self.checkpoint, checkpoint_path)

    def update_checkpoint(self):
        self.checkpoint['netE_state'] = self.models['netE'].state_dict()
        self.checkpoint['optE_state'] = self.opts['optE'].state_dict()
        for task in self.args.task_types:
            self.checkpoint['net_{}_state'.format(task)] = self.models[task].state_dict()
            self.checkpoint['opt_{}_state'.format(task)] = self.opts[task].state_dict()
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
        if losses is not None:
            loss_message = ', '.join(['[{}_{}]:{:.4f}'.format(mode, key, loss.item()) for key, loss in losses.items()])
            message = result_message + 'Losses:' + loss_message
        if evaluations is not None:
            evaluations_message = ', '.join(['[{}_{}]:{:.4f}'.format(mode, key, evaluation) for key, evaluation in evaluations.items()])
            message = message + '; Evaluations:' + evaluations_message
        return message
    '''
        Save results
    '''
    def save_results(self):
        pass

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

    '''
        Update Log and update tensorboard image results
    '''
    def update_stroke_results(self, mode, strokes, strokes_pred, masks, size=128):
        imgs = []
        imgs_incomp = []
        imgs_pred = []
        strokes_pred_type = strokes_pred[:,:,2:5]
        if 'maskgmm' in self.args.task_types:
            pis, mus, sigmas, rhos, qs = self.decode_parameters(strokes_pred)
            strokes_pred = self.sample_stroke(pis, mus, sigmas, rhos, qs, self.args.gamma)
        elif 'maskrec' in self.args.task_types:
            strokes_pred[:,:,2:5] = strokes_pred_type == strokes_pred_type.max(dim=2, keepdim=True)[0]
        strokes_pred = strokes_pred * (1-masks) + strokes * masks
        for i in range(min(len(strokes), self.args.num_val_samples)):
            if self.args.stroke_type == 'stroke-discrete':
                tmp_stroke = disc2stroke5(strokes[i].cpu().numpy(), self.args.max_size)
                tmp_stroke_pred = disc2stroke5(strokes_pred[i].cpu().numpy(), self.args.max_size)
                tmp_stroke_incomp = rec_incomplete_strokes(disc2stroke5(strokes[i].cpu().numpy(), self.args.max_size), masks[i].cpu().numpy())
            else:
                tmp_stroke = strokes[i].cpu().numpy()
                tmp_stroke_pred = strokes_pred[i].cpu().numpy()
                #)
                tmp_stroke_incomp = rec_incomplete_strokes(strokes[i].cpu().numpy(), masks[i].cpu().numpy())
            if self.args.stroke_type == 'stroke-5' or self.args.stroke_type == 'stroke-discrete':
                tmp_stroke = to_normal_strokes(tmp_stroke)
                tmp_stroke_pred = to_normal_strokes(tmp_stroke_pred)
                tmp_stroke_incomp = to_normal_strokes(tmp_stroke_incomp)
            imgs.append(strokes2drawing(tmp_stroke,  size=128, svg_filename='{}/tmp_{}.svg'.format(self.tmp_dir, i)))
            imgs_incomp.append(strokes2drawing(tmp_stroke_incomp,  size=128, svg_filename='{}/tmp_incomp_{}.svg'.format(self.tmp_dir, i)))
            imgs_pred.append(strokes2drawing(tmp_stroke_pred,  size=128, svg_filename='{}/tmp_pred_{}.svg'.format(self.tmp_dir, i)))

        #[:self.args.num_display_samples]
        imgs, imgs_incomp, imgs_pred = np.array(imgs), np.array(imgs_incomp), np.array(imgs_pred)
        #print(imgs.shape, imgs_pred.shape)
        record_imgs = np.concatenate([imgs, imgs_incomp, imgs_pred], axis=2)
        for i in range(len(record_imgs)):
            #print(i)
            self.tensorboard_logger.add_image('{}_record_imgs_{}'.format(mode,i), record_imgs[i].reshape((1,)+record_imgs[i].shape), self.counters['t'])
