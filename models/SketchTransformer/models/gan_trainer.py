import os
import time
import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

from util.evaluation import AverageMeter, accuracy, pairwise_distances
from trainer.trainer import Trainer
from modules.losses import get_gan_loss, KLLoss
from .networks import SketchTransformerModel, MaskSketchRecModel, MaskSketchGMMModel, SketchClassificationModel, SketchGANGenerator, SketchGANDiscriminator, GMMLoss, KLLoss
from .utils import strokes2drawing, to_normal_strokes

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical

TASKMODELS = {'maskrec':MaskSketchRecModel, 'maskgmm':MaskSketchGMMModel, 'sketchcls':SketchClassificationModel}

class SketchTransformerGANTrainer(Trainer):
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
        netG_kwargs,netD_kwargs = self.build_models(args)
        self.counters = {'t':1, 'epoch':1, 'now_t':1}
        self.iter_per_epoch = datas['iter_per_epoch']
        assert self.args.noise_type in ['full', 'single']
        self.checkpoint = self.initialize_checkpoint(args, netG_kwargs,netD_kwargs)

        self.running_paras = {'output_attentions':self.args.output_attentions, 'output_all_states':self.args.output_all_states, 'keep_multihead_output':self.args.keep_multihead_output}
        tmp_dir = os.path.join('./tmp', self.args.log_id)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir

    def build_models(self, args):
        self.logger.info('Defining Network Structure...')
        # Generator
        netG_kwargs = {
            'layers_setting':args.gen_layers_setting, 'input_dim': args.noise_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length,
            'position_type':args.gen_position_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        netG = SketchGANGenerator(**netG_kwargs)

        # Discriminator
        netD_kwargs = {
            'layers_setting':args.dis_layers_setting, 'input_dim': args.input_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length+1,
            'position_type':args.dis_position_type, 'attention_norm_type':args.attention_norm_type, 'inter_activation':args.inter_activation, 'attention_dropout_prob':args.attention_dropout_prob,
            'hidden_dropout_prob':args.hidden_dropout_prob, 'output_dropout_prob':args.output_dropout_prob
        }

        netD = SketchGANDiscriminator(**netD_kwargs)

        self.logger.info("Sketch Transformer Sturture:\n{}".format(netG))

        self.models = {'netG':netG,  'netD':netD}

        self.gpu_ids = {'netG':torch.device('cuda:{}'.format(args.gpu_ids[0])),
                        'netD':torch.device('cuda:{}'.format(args.gpu_ids[0])),}

        TASKPARAS = {'maskrec':{'hidden_dim':args.hidden_dim, 'input_dim':args.input_dim},
                     'maskgmm':{'hidden_dim':args.hidden_dim, 'M':args.M},
                     'sketchcls':{'hidden_dim':args.hidden_dim, 'cls_dim':args.cls_dim, 'max_length':args.max_length}}

        for task in args.task_types:
            self.models[task] = TASKMODELS[task](**TASKPARAS[task])
            self.gpu_ids[task] = torch.device('cuda:{}'.format(args.gpu_ids[0]))

        # Change to gpu
        for key in self.models:
            self.models[key] = self.models[key].to(self.gpu_ids[key])

        self.logger.info('Finish Construction of Networks.')

        # Defining Optimizers
        self.opts = {
            'optG': torch.optim.Adam(netG.parameters(), lr=args.learning_rate),
            'optD': torch.optim.Adam(netD.parameters(), lr=args.learning_rate),
        }

        for task in args.task_types:
            self.opts[task] = torch.optim.Adam(self.models[task].parameters(), lr=args.learning_rate)


        self.logger.info('Finish Construction of Optimizers.')

        # Defining losses
        self.logger.info('Defining Losses...')
        gan_g_loss, gan_d_loss = get_gan_loss('gan')
        self.losses = {'gan_gen': gan_g_loss, 'gan_dis':gan_d_loss, 'mask_axis':F.l1_loss, 'mask_type':nn.CrossEntropyLoss(), 'prediction':nn.CrossEntropyLoss(),}

        self.losses_weights = {'gan_gen': args.gan_gen_weight, 'gan_dis':args.gan_dis_weight, 'mask_axis':args.mask_axis_weight, 'mask_type':args.mask_type_weight, 'prediction':args.prediction_weight}
        self.valid_losses = [key for key in self.losses_weights if self.losses_weights[key] > 0]
        self.logger.info('Finish Losses.')

        return netG_kwargs, netD_kwargs

    def initialize_checkpoint(self, args, netG_kwargs, netD_kwargs):
        checkpoint = {
            'args': args.__dict__,
            'netG_kwargs': netG_kwargs,
            'netD_kwargs': netD_kwargs,
            'counters': {
                't': None,
                'epoch': None,
            },
            'netD_state': None, 'netD_best_state': None, 'optD_state': None,
            'netG_state': None, 'netG_best_state': None, 'optG_state': None,
            'best_t': [],
        }
        return checkpoint

    def get_random_input(self, batch_size, seq_len, noise_dim, device):
        if self.args.noise_type == 'full':
            noise = torch.randn(batch_size, seq_len, noise_dim, device=device)
        elif self.args.noise_type == 'single':
            noise = torch.randn(batch_size, 1, noise_dim, device=device)
            noise = noise.repeat(1, seq_len, 1)
        return noise
    '''
        Training: only for training process.
        input:
            None: Take From Trainer Itself.
    '''
    def train(self):
        netG, netD = self.models['netG'],self.models['netD']
        optG, optD = self.opts['optG'],self.opts['optD']

        for epoch_i in range(self.args.num_epoch):
            start_time = time.time()
            for data_i, batch_data in enumerate(self.datas['train_loader']):
                end_time = time.time()
                self.data_time = end_time - start_time
                self.counters['now_t'] = data_i + 1

                # Tranining Process
                batch_data = [tensor.to(device=self.gpu_ids['netG'],dtype=torch.float32) for tensor in batch_data]
                mask_input_states, input_states, length_mask, input_mask, target = batch_data
                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]

                target = target.to(dtype=torch.long)
                # Fake Sequence Generation
                noise_input = self.get_random_input(input_states.size(0), input_states.size(1), self.args.noise_dim, device=self.gpu_ids['netG'])

                fake_states = netG(noise_input, attention_mask=None, head_mask=None, **self.running_paras)

                # Add gan label for input
                prefix_input = torch.ones(input_states.size(0), 1, input_states.size(2)).to(dtype=torch.float, device=self.gpu_ids['netG'])
                dis_input_states = torch.cat([prefix_input*(-1), input_states], dim=1)
                dis_fake_states = torch.cat([prefix_input*(-1), fake_states], dim=1)

                # compute real and fake scores
                scores_real = netD(dis_input_states, attention_mask=None, head_mask=None, **self.running_paras)
                scores_fake = netD(dis_fake_states, attention_mask=None, head_mask=None, **self.running_paras)

                # Update the Discriminator
                # Size Checking: scores_real, scores_fake [batch, 2]
                total_d_loss, d_losses = self.calculate_d_losses(scores_real, scores_fake)
                # Optimizing
                optD.zero_grad()
                total_d_loss.backward(retain_graph=True)
                optD.step()

                # Update the Generator
                scores_fake = netD(dis_fake_states, attention_mask=None, head_mask=None, **self.running_paras)
                total_g_loss, g_losses = self.calculate_g_losses(scores_fake)

                # Optimizing
                optG.zero_grad()
                total_g_loss.backward()
                optG.step()

                # Update results
                self.run_time = time.time() - end_time
                self.batch_time = self.data_time + self.run_time

                if self.counters['t'] % self.args.print_every == 0:
                    evaluations = self.evaluate()
                    self.update_log('train',{ **g_losses, **d_losses}, evaluations)

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
        netG, netD = self.models['netG'], self.models['netD']
        total_losses = {'gan_gen':AverageMeter(), 'gan_dis':AverageMeter()}

        total_evaluations = {'No':AverageMeter()}
        topk = (1,5)
        with torch.no_grad():
            for data_i, batch_data in enumerate(self.datas['val_loader']):
                batch_data = [tensor.to(device=self.gpu_ids['netG'],dtype=torch.float32) for tensor in batch_data]
                mask_input_states, input_states, length_mask, input_mask, target = batch_data
                # Size Checking: input_states[batch, seq_len, 5], length_mask[batch,seq_len], input_mask[batch, seq_len], target[batch]

                target = target.to(dtype=torch.long)

                # Fake Sequence Generation
                noise_input = self.get_random_input(input_states.size(0), input_states.size(1), self.args.noise_dim, device=self.gpu_ids['netG'])

                fake_states = netG(noise_input, attention_mask=None, head_mask=None, **self.running_paras)

                # Add gan label for input
                prefix_input = torch.ones(input_states.size(0), 1, input_states.size(2)).to(dtype=torch.float, device=self.gpu_ids['netG'])
                dis_input_states = torch.cat([prefix_input*(-1), input_states], dim=1)
                dis_fake_states = torch.cat([prefix_input*(-1), fake_states], dim=1)

                # compute real and fake scores
                scores_real = netD(dis_input_states, attention_mask=None, head_mask=None, **self.running_paras)
                scores_fake = netD(dis_fake_states, attention_mask=None, head_mask=None, **self.running_paras)

                # Update the Discriminator
                # Size Checking: scores_real, scores_fake [batch, 2]
                total_d_loss, d_losses = self.calculate_d_losses(scores_real, scores_fake)

                # Update the Generator
                scores_fake = netD(dis_fake_states, attention_mask=None, head_mask=None, **self.running_paras)
                total_g_loss, g_losses = self.calculate_g_losses(scores_fake)

                evaluations = self.evaluate()
                losses = {**d_losses, **g_losses}
                for key in total_losses:
                    total_losses[key].update(losses[key].item(), input_states.size(0))
                for key in total_evaluations:
                    total_evaluations[key].update(evaluations[key], input_states.size(0))

        log_losses = {key:torch.tensor(total_losses[key].avg) for key in total_losses}
        log_evaluations = {key:total_evaluations[key].avg for key in total_evaluations}
        # Update Logs
        self.update_log('val', log_losses, log_evaluations)
        strokes_pred = fake_states
        self.update_stroke_results('val', input_states, strokes_pred)


    def decode_parameters(self, y):
        # y [batch, seq_len, 6*M+3]
        M = int((y.size(2)-3)/6)
        g_paras = y[:, :, :6*M].view(y.size(0), y.size(1), M, 6)
        pis = g_paras[:,:,:, 0]
        mus = torch.exp(g_paras[:,:,:, 1:3])
        sigmas = torch.exp(g_paras[:,:,:, 3:5])
        rhos = torch.tanh(g_paras[:,:,:, 5])
        qs = y[:,:, 6*M:6*M+3]

        return pis, mus, sigmas, rhos, qs

    def sample_stroke(self, pis, mus, sigmas, rhos, qs, gamma):
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
        strokes = []
        sigmas = sigmas * gamma
        # Sample for each sketch
        for i in range(batch_size):
            #print(pis[:,i,:].size(), pis[:,i,:].device)
            #print(pis.size(), mus.size(), sigmas.size(), rhos.size(), qs.size())
            comp_m = OneHotCategorical(logits=pis[i, :, :])
            comp_choice = (comp_m.sample()==1)

            mu, sigma, rho, q = mus[i,:,:,:][comp_choice], sigmas[i,:,:,:][comp_choice], rhos[i,:,:][comp_choice], qs[i,:,:]

            cov = torch.stack([torch.diag(sigma[j]*sigma[j]) + (1-torch.eye(2).to(mu.device)) * rho[j] * torch.prod(sigma[j]) for j in range(seq_len)]).to(device=mu.device)


            normal_m = MultivariateNormal(mu, cov)
            stroke_move = normal_m.sample().to(pis.device) # [seq_len, 2]
            pen_states = (q == q.max(dim=1, keepdim=True)[0]).to(dtype=torch.float)#[seq_len, 3]
            stroke = torch.cat([stroke_move, pen_states], dim=1).to(pis.device)
            strokes.append(stroke)

        return torch.stack(strokes)

    def calculate_g_losses(self, score_fake):
        # Intialize total_loss and losses
        losses = {}
        total_loss = torch.zeros(1).to(self.gpu_ids['netG'])

        losses['gan_gen'] = self.losses_weights['gan_gen'] * self.losses['gan_gen'](score_fake)

        total_loss = total_loss + losses['gan_gen']

        return total_loss, losses

    def calculate_d_losses(self, score_real, score_fake):
        # Intialize total_loss and losses
        losses = {}
        total_loss = torch.zeros(1).to(self.gpu_ids['netG'])

        losses['gan_dis'] = self.losses_weights['gan_dis'] * self.losses['gan_dis'](score_real, score_fake)

        total_loss = total_loss + losses['gan_dis']
        return total_loss, losses

    # def evaluate(self, prediction, target, topk=(1,5)):
    #     results = {}
    #     #print(prediction, target)
    #     res_acc = accuracy(prediction, target, topk=topk)
    #     for k in topk:
    #         results['accuracy_{}'.format(k)] = res_acc[k][0]
    #
    #     return results

    def evaluate(self, topk=(1,5)):
        results = {'No':1}
        return results

    def restore_models(self):
        netG, netD = self.models['netG'], self.models['netD']
        optG, optD = self.opts['optG'], self.opts['optD']
        restore_path = self.args.restore_checkpoint_path
        logger.info('Restoring from checkpoint:{}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        if 'gen' in self.args.load_pretrained:
            netG.load_state_dict(checkpoint['netG_state'])
            optG.load_state_dict(checkpoint['optG_state'])

        if 'dis' in self.args.load_pretrained:
            netD.load_state_dict(checkpoint['netD_state'])
            optD.load_state_dict(checkpoint['optD_state'])

        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        return t, epoch, checkpoint

    def save_checkpoint(self, name):
        checkpoint_path = '{}/{}_ckpt.pth.tar'.format(self.log_dir, name)
        self.logger.info('Saving checkpoint to {}'.format(checkpoint_path) )
        torch.save(self.checkpoint, checkpoint_path)

    def update_checkpoint(self):
        self.checkpoint['netG_state'] = self.models['netG'].state_dict()
        self.checkpoint['optG_state'] = self.opts['optG'].state_dict()
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
        for key, loss in losses.items():
            self.tensorboard_logger.scalar_summary('{}_{}'.format(mode, key), loss.item(), self.counters['t'])

        for key, evaluation in evaluations.items():
            self.tensorboard_logger.scalar_summary('{}_{}'.format(mode, key), evaluation, self.counters['t'])

    '''
        Update Log and update tensorboard image results
    '''
    def update_stroke_results(self, mode, strokes, strokes_pred, size=128):
        imgs = []
        imgs_pred = []
        strokes_pred_type = strokes_pred[:,:,2:5]
        if 'maskgmm' in self.args.task_types:
            pis, mus, sigmas, rhos, qs = self.decode_parameters(strokes_pred)
            strokes_pred = self.sample_stroke(pis, mus, sigmas, rhos, qs, self.args.gamma)
        else:
            strokes_pred[:,:,2:5] = strokes_pred_type == strokes_pred_type.max(dim=2, keepdim=True)[0]

        for i in range(min(len(strokes), self.args.num_val_samples)):
            tmp_stroke = strokes[i].cpu().numpy()
            tmp_stroke_pred = strokes_pred[i].cpu().numpy()
            if self.args.stroke_type == 'stroke-5':
                tmp_stroke = to_normal_strokes(tmp_stroke)
                tmp_stroke_pred = to_normal_strokes(tmp_stroke_pred)
            imgs.append(strokes2drawing(tmp_stroke,  size=128, svg_filename='{}/tmp_{}.svg'.format(self.tmp_dir, i)))
            imgs_pred.append(strokes2drawing(tmp_stroke_pred,  size=128, svg_filename='{}/tmp_pred_{}.svg'.format(self.tmp_dir, i)))
        #[:self.args.num_display_samples]
        imgs, imgs_pred = np.array(imgs), np.array(imgs_pred)
        #print(imgs.shape, imgs_pred.shape)
        record_imgs = np.concatenate([imgs, imgs_pred], axis=2)

        self.tensorboard_logger.image_summary('{}_record_imgs'.format(mode), record_imgs, self.counters['t'])
