import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gan_loss(gan_type):
    '''
        Work as a factory function
    '''
    if gan_type == 'gan':
        return GANGLoss(), GANDLoss()
    elif gan_type == 'hinge_gan':
        return HingeGANGLoss(), HingeGANDLoss()

class KLLoss(nn.Module):
    """
        With a standard gaussian noise.
    """
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, sigma):
        return torch.mean(0.5 * torch.sum(torch.exp(sigma) + mu**2 - 1. - sigma, 1))

'''
Original GAN Loss terms, for generator and discriminator.
'''
class GANGLoss(nn.Module):
    def __init__(self):
        super(GANGLoss, self).__init__()
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
    def forward(self, scores_fake):
        scores_fake = scores_fake.view(-1)
        y_fake = torch.full_like(scores_fake, 1)
        return self.bce_logit_loss(scores_fake, y_fake)

class GANDLoss(nn.Module):
    def __init__(self):
        super(GANDLoss, self).__init__()
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
    def forward(self, scores_real, scores_fake):
        scores_real = scores_real.view(-1)
        scores_fake = scores_fake.view(-1)
        y_real = torch.full_like(scores_real, 1)
        y_fake = torch.full_like(scores_fake, 0)
        loss_real = self.bce_logit_loss(scores_real, y_real)
        loss_fake = self.bce_logit_loss(scores_fake, y_fake)
        return loss_real + loss_fake


'''
Hinge GAN Loss terms, for generator and discriminator.
'''
class HingeGANGLoss(nn.Module):
    def __init__(self):
        super(HingeGANGLoss, self).__init__()

    def forward(self, scores_fake):
        scores_fake = scores_fake.view(-1)
        return -scores_fake.mean()

class HingeGANDLoss(nn.Module):
    def __init__(self):
        super(HingeGANDLoss, self).__init__()

    def forward(self, scores_real, scores_fake):
        scores_real = scores_real.view(-1)
        scores_fake = scores_fake.view(-1)
        return F.relu(1-scores_real).mean() + F.relu(1+scores_fake).mean()
