import os
import time
import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

from util.evaluation import AverageMeter, accuracy, pairwise_distances
from trainer.trainer import Trainer
from modules.losses import get_gan_loss, KLLoss
from .networks import SketchTransformerModel, SketchCNN, MaskSketchRecModel, MaskSketchGMMModel, SketchClassificationModel, SketchClsPoolingModel, SketchRetrievalPoolingModel, SketchDiscretePoolingModel, GMMLoss
from .utils import strokes2drawing, to_normal_strokes, rec_incomplete_strokes, disc2stroke5

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical

TASKMODELS = {'sbir':SketchRetrievalPoolingModel, 'sbir_cls':nn.Linear}

class SketchTransformerSBIRTrainer(Trainer):
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
        self.cls_in_input = 'sketchclsinput' in args.task_types or 'sbir_cls' in args.task_types
        self.rel_in_input = 'sketchretrieval' in args.task_types or 'sbir' in args.task_types
        netSE_kwargs = self.build_models(args)
        self.counters = {'t':1, 'epoch':1, 'now_t':1}
        self.iter_per_epoch = datas['iter_per_epoch']
        self.checkpoint = self.initialize_checkpoint(args, netSE_kwargs)
        self.best_t = 0
        self.best_metric = 0.0
        self.cate_num = self.datas['val_loader'].dataset.get_cate_num()
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
        netSE_kwargs = {
            'model_type':args.model_type,'layers_setting':args.layers_setting, 'embed_layers_setting':args.embed_layers_setting, 'input_dim': args.input_dim, 'max_length':args.max_length+self.cls_in_input+self.rel_in_input, 'max_size':args.max_size, 'type_size':args.type_size,
            'position_type':args.position_type, 'segment_type':args.segment_type, 'sketch_embed_type':args.sketch_embed_type, 'embed_pool_type':args.embed_pool_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        # Sketch Encoder
        netSE = SketchTransformerModel(**netSE_kwargs)


        # Image Encoder
        netIE = SketchCNN(args.rel_feat_dim, args.image_net_type, args.cnn_pretrained)
        self.models = {'netSE':netSE, 'netIE':netIE}

        self.logger.info("Sketch Transformer Sturture:\n{}\n{}".format(netSE, netIE))


        self.gpu_ids = {'netSE':torch.device('cuda:{}'.format(args.gpu_ids[0])),
                        'netIE':torch.device('cuda:{}'.format(args.gpu_ids[0])),}

        TASKPARAS = {'sbir_cls':{'in_features':args.rel_feat_dim, 'out_features':args.cls_dim},
            'sbir':{'rel_layers_setting':args.rel_layers_setting, 'hidden_dim':args.layers_setting[0][1], 'feat_dim':args.rel_feat_dim, 'pool_dim':0}}

        for task in args.task_types:
            self.models[task] = TASKMODELS[task](**TASKPARAS[task])
            self.gpu_ids[task] = torch.device('cuda:{}'.format(args.gpu_ids[0]))

        # Change to gpu
        for key in self.models:
            self.models[key] = self.models[key].to(self.gpu_ids[key])

        self.logger.info('Finish Construction of Networks.')

        # Defining Optimizers
        self.opts = {
            'optSE': torch.optim.Adam(netSE.parameters(), lr=args.learning_rate),
            'optIE': torch.optim.Adam(netIE.parameters(), lr=args.learning_rate)
        }


        for task in args.task_types:
            self.opts[task] = torch.optim.Adam(self.models[task].parameters(), lr=args.learning_rate)


        self.logger.info('Finish Construction of Optimizers.')

        # Defining losses
        self.logger.info('Defining Losses...')
        self.losses = {'prediction':nn.CrossEntropyLoss(),'triplet':nn.TripletMarginLoss(margin=args.triplet_margin)}

        self.losses_weights = {'prediction':args.prediction_weight, 'triplet':args.triplet_weight}
        self.valid_losses = [key for key in self.losses_weights if self.losses_weights[key] > 0]
        self.logger.info('Finish Losses.')

        return netSE_kwargs

    def initialize_checkpoint(self, args, netSE_kwargs):
        checkpoint = {
            'args': args.__dict__,
            'netSE_kwargs': netSE_kwargs,
            'counters': {
                't': None,
                'epoch': None,
            },
            'netSE_state': None, 'netSE_best_state': None, 'optSE_state': None,
            'netIE_state': None, 'netIE_best_state': None, 'optIE_state': None,
            'best_t': None,
        }
        for task in args.task_types:
            checkpoint['net_{}_state'.format(task)] = None
            checkpoint['net_{}_best_state'.format(task)] = None
            checkpoint['opt_{}_state'.format(task)] = None
        return checkpoint

    def restore_models(self):
        netSE, netIE = self.models['netSE'], self.models['netIE']
        optSE, optIE = self.opts['optSE'],self.opts['optIE']
        restore_path = self.args.restore_checkpoint_path
        self.logger.info('Restoring from checkpoint:{}'.format(restore_path))
        checkpoint = torch.load(restore_path, map_location=self.gpu_ids['netSE'])

        if 'sketch_enc_net_from_mask' in self.args.which_pretrained:
            netSE.load_model(checkpoint['netE_state'], self.rel_in_input, self.cls_in_input, 'sketchretrieval' in checkpoint['args']['_cfg_dict']['task_types'], 'sketchclsinput' in checkpoint['args']['_cfg_dict']['task_types'])
            self.logger.info('Restoring netE')

        if 'sketch_enc_net' in self.args.which_pretrained:
            netSE.load_model(checkpoint['netSE_state'], self.rel_in_input, self.cls_in_input, 'sketchretrieval' in checkpoint['args']['_cfg_dict']['task_types'], 'sketchclsinput' in checkpoint['args']['_cfg_dict']['task_types'])
            self.logger.info('Restoring netSE')
        if 'image_enc_net' in self.args.which_pretrained:
            netIE.load_state_dict(checkpoint['netIE_state'])
            self.logger.info('Restoring netIE')
        if 'task_net' in self.args.which_pretrained:
            for task in self.args.task_types:
                if 'net_{}_state'.format(task) in checkpoint:
                    self.models[task].load_state_dict(checkpoint['net_{}_state'.format(task)])
            self.logger.info('Restoring net tasks')
        # if 'enc_opt' in self.args.which_pretrained:
        #     optE.load_state_dict(checkpoint['optE_state'])
        # if 'task_opt' in self.args.which_pretrained:
        #     for task in self.args.task_types:
        #         if 'opt_{}_state'.format(task) in checkpoint:
        #             self.opts[task].load_state_dict(checkpoint['opt_{}_state'.format(task)])

        if 'continue' == self.args.load_pretrained:
            self.checkpoint['counters']['t'] = checkpoint['counters']['t']
            self.checkpoint['counters']['epoch'] = checkpoint['counters']['epoch']

    '''
    Training process for retrieval task
    '''
    def train(self):
        netSE, netIE = self.models['netSE'], self.models['netIE']
        optSE, optIE = self.opts['optSE'], self.opts['optIE']

        for epoch_i in range(self.args.num_epoch):
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['train_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time
                self.counters['now_t'] = data_i + 1


                batch_data = [tensor.to(device=self.gpu_ids['netSE'],dtype=torch.float32)  for tensor in batch_data]
                strokes, pos_images, neg_images, segments, length_masks,  targets = batch_data

                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segments = segments.to(dtype=torch.long)
                targets = targets.to(dtype=torch.long)
                # Sketch feature
                output_states = netSE(strokes, length_masks, segments=segments, head_mask=None, **self.running_paras)
                sketch_states = self.models['sbir'](output_states) # [batch, rel_feat_dim]
                sketch_states = F.normalize(sketch_states)
                # Image Feature
                pos_image_states = netIE(pos_images)
                neg_image_states = netIE(neg_images)
                pos_image_states, neg_image_states = F.normalize(pos_image_states), F.normalize(neg_image_states)
                cls_states = None
                if 'sbir_cls' in self.args.task_types:
                    cls_states = self.models['sbir_cls'](torch.cat([sketch_states, pos_image_states], dim=0))
                    targets = torch.cat([targets, targets], dim=0)
                # Size Checking: output_states, pooled_output[[batch, seq_len, 6*M+3],[]]
                total_loss, losses = self.calculate_losses(sketch_states, pos_image_states, neg_image_states, cls_states, targets)

                # Optimizing
                optSE.zero_grad(), optIE.zero_grad()
                for task in self.args.task_types:
                    self.opts[task].zero_grad()
                total_loss.backward()
                optSE.step(), optIE.step()
                for task in self.args.task_types:
                    self.opts[task].step()
                # Update results
                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

                if self.counters['t'] % self.args.print_every == 0:
                    evaluations = None
                    # self.evaluate(sketch_states, pos_, targets)
                    self.update_log('train', losses, evaluations)

                if self.counters['t'] % self.args.checkpoint_every == 0:
                    #self.update_img_results('train', imgs, imgs_pred[-1], segs_image)
                    self.validate()
                    self.update_checkpoint()
                    self.save_checkpoint('latest')
                    if self.best_t == self.counters['t']:
                        self.save_checkpoint('best')

                if self.counters['t'] % self.args.save_model_every == 0:
                    self.save_checkpoint('iter_{}'.format(self.counters['t']))
                self.counters['t'] += 1
                start_time = time.time()
            self.counters['epoch'] += 1

    def validate(self):
        netSE, netIE = self.models['netSE'], self.models['netIE']
        total_losses = self.initialize_validate_losses()
        topk = (1,5)
        total_evaluations = {'accuracy_{}'.format(k): AverageMeter() for k in topk}
        total_sketch_states,total_image_states, total_sketch_classes, total_image_classes = [], [], [], []
        # Record the sketch states[batch, rel_feat_dim]
        with torch.no_grad():
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['val_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time

                batch_data = [tensor.to(device=self.gpu_ids['netSE'],dtype=torch.float32)  for tensor in batch_data]
                strokes, pos_images, neg_images, segments, length_masks,  targets = batch_data

                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]
                segments = segments.to(dtype=torch.long)
                targets = targets.to(dtype=torch.long)
                # Sketch feature
                output_states = netSE(strokes, length_masks, segments=segments, head_mask=None, **self.running_paras)

                sketch_states = self.models['sbir'](output_states) # [batch, rel_feat_dim]
                sketch_states = F.normalize(sketch_states)
                total_sketch_states.append(sketch_states)
                total_sketch_classes.append(targets)
                # Size Checking: output_states, pooled_output[[batch, seq_len, 6*M+3],[]]

                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

        image_loader = self.datas['val_loader'].dataset.get_image_loader(shuffle=False)
        # Record the Image states[batch, rel_feat_dim]
        with torch.no_grad():
            start_time = time.time()
            for data_i, batch_data in enumerate(image_loader):
                end_time = time.time()
                self.data_time = end_time - start_time

                batch_data = [tensor.to(device=self.gpu_ids['netSE'],dtype=torch.float32)  for tensor in batch_data]
                images, targets = batch_data

                targets = targets.to(dtype=torch.long)
                # Sketch feature
                image_states = netIE(images)
                image_states = F.normalize(image_states)
                total_image_states.append(image_states)
                total_image_classes.append(targets)

                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

        sbir_evaluations = {}
        sbir_evaluations = self.retrieval_evaluation(torch.cat(total_sketch_states, dim=0),torch.cat(total_sketch_classes, dim=0),torch.cat(total_image_states, dim=0),torch.cat(total_image_classes, dim=0), topk=topk)
        log_losses = {key:torch.tensor(total_losses[key].avg) for key in total_losses}
        log_evaluations = sbir_evaluations
        if self.best_metric <= log_evaluations['retrieval_1']:
            self.best_metirc = log_evaluations['retrieval_1']
            self.best_t = self.counters['t']
        # Update Logs
        self.update_log('val', log_losses, log_evaluations)

    def calculate_losses(self, sketch_states, pos_image_states, neg_image_states, cls_states, targets):
        # Intialize total_loss and losses
        losses = {}
        total_loss = torch.zeros(1).to(self.gpu_ids['netSE'])


        if 'sbir' in self.args.task_types:
            losses['triplet'] = self.losses_weights['triplet'] * self.losses['triplet'](sketch_states, pos_image_states, neg_image_states)
            total_loss = total_loss + losses['triplet']

        if 'sbir_cls' in self.args.task_types and cls_states is not None:
            #print(cls_states.size(), targets.size())
            losses['prediction'] = self.losses_weights['prediction'] * self.losses['prediction'](cls_states, targets)
            total_loss = total_loss + losses['prediction']
        return total_loss, losses

    def retrieval_evaluation(self, test_states, test_targets, collection_states, collection_targets, topk=(1,5), batch_size=32):
        #test_states, test_targets, collection_states, collection_targets = test_states.cpu(), test_targets, collection_states, collection_targets,
        num_test_batch = test_states.size(0) // batch_size + int(test_states.size(0) % batch_size != 0)
        num_collect_batch = collection_states.size(0) // batch_size + int(collection_states.size(0) % batch_size != 0)
        correct = {k:0 for k in  topk}
        MAPs = []
        for test_i in range(num_test_batch):
            test_i_distances = []
            for collect_i in range(num_collect_batch):
                test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
                collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
                tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
                test_i_distances.append(tmp_distances)
            test_i_distances = torch.cat(test_i_distances, dim=1) #[batch, length]
            #print(test_i_distances.size())
            _ , pred_indics = test_i_distances.sort(dim=1)
            #test_indics = [i for i in range(test_i*batch_size, (test_i+1)*batch_size)]
            #print(test_indics, test_targets)

            test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]
            # pred_i_target = collection_targets[pred_indics]
            for j in range(len(test_i_target)):
                for k in topk:
                    if test_i_target[j] in collection_targets[pred_indics[j][:k]]:
                        correct[k] = correct[k] + 1

            # pred_i_target = collection_targets[pred_indics]
            for j in range(len(test_i_target)):
                c = test_i_target[j]
                res = (collection_targets[pred_indics[j]] == c)
                #print(collection_targets[pred_indics[j]], res)
                k, rightk, precision = 0, 0, []
                while rightk < self.cate_num[c.item()]:
                    r = res[k].item()
                    if r:
                        precision.append((res[:k + 1]).to(torch.float).mean().item())
                        rightk += 1
                    k += 1
                #print(precision)
                MAPs.append(sum(precision) / len(precision))
        MAP = sum(MAPs) / len(MAPs)
        results = {'retrieval_{}'.format(k):(correct[k] / test_states.size(0)) for k in topk}
        return {**results, 'map':MAP}


    def initialize_validate_losses(self):
        total_losses = {}
        if 'sbir' in self.args.task_types:
            total_losses['triplet'] = AverageMeter()
        if 'sbir_cls' in self.args.task_types:
            total_losses['prediction'] = AverageMeter()
        return total_losses


    def save_checkpoint(self, name):
        checkpoint_path = '{}/{}_ckpt.pth.tar'.format(self.log_dir, name)
        self.logger.info('Saving checkpoint to {}'.format(checkpoint_path) )
        torch.save(self.checkpoint, checkpoint_path)

    def update_checkpoint(self):
        self.checkpoint['netSE_state'] = self.models['netSE'].state_dict()
        self.checkpoint['optSE_state'] = self.opts['optSE'].state_dict()
        self.checkpoint['netIE_state'] = self.models['netIE'].state_dict()
        self.checkpoint['optIE_state'] = self.opts['optIE'].state_dict()
        for task in self.args.task_types:
            self.checkpoint['net_{}_state'.format(task)] = self.models[task].state_dict()
            self.checkpoint['opt_{}_state'.format(task)] = self.opts[task].state_dict()
        self.checkpoint['best_t'] = self.best_t
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
