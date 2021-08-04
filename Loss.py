import time
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from  vgg19 import *
import random
#from dataloader import celebA as DA
#from dataloader import preprocess as preprocess
from dataloader import Voxall as DA
from models import *
import pytorch_ssim
#from Face_model import Backbone

'''
device = torch.device('cuda:0')
model = Backbone(50, 0.6, 'ir_se').to(device)
state_dictc = torch.load('./model_ir_se50.pth')
model.load_state_dict(state_dictc)
model.eval()

def face_emb(x):
    x = x.clone()
    x = F.upsample(x, size = (112,112),mode='bilinear', align_corners = False)
    x = (x-0.5)/0.5
    embedding = model(x)
    return embedding
'''

def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


def photometric_loss(im1, im2, conf_sigma=None, mask=None):
    loss = (im1-im2).abs()
    if conf_sigma is not None:
        loss = loss *2**0.5 / (conf_sigma +1e-7) + (conf_sigma + 1e-7).log()
    if mask is not None:
        mask = mask.expand_as(loss)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss


def ssim_loss(x,y):
    return (1.0-pytorch_ssim.SSIM()(x,y))

class TotalVaryLoss(nn.Module):
    def __init__(self):
        super(TotalVaryLoss, self).__init__()

    def forward(self, x, weight=1):
        loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return 0.00001*loss.mean()
        
def margin_loss(x,y, margin=0.1):
    l1 = (x-y).abs()-0.1
    l1 - F.relu(l1)
    return l1.mean()

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G
        
def perceptual_loss(input, target):
    out = nn.MSELoss()(GramMatrix()(input), GramMatrix()(target).detach())
    return out
    
def L2loss(embed1, embed2):
    dist = torch.norm((embed1-embed2), p=2, dim=1)
    dist, _ = torch.max(dist-0.1, 0)
    return dist.mean() 

def symetricloss(out0):
    b,c,h,w = out0.size()
    left_face = out0[:,:,:,:(w//2)].contiguous()     
    right_face = torch.flip(out0[:,:,:,(w//2):],dims=[3]).contiguous() 
    return photometric_loss(left_face, right_face, conf_sigma=None)    


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    def forward(self, X):
        error = torch.sqrt(X**2 + self.eps )
        return error.mean()


def _ternary_transform(image):
    patch_size = 3
    b,c,h,w = image.size()
    intensities = 0.299*image[:,0,:,:] + 0.857*image[:,1,:,:] + 0.114*image[:,2,:,:]
    intensities = intensities.view(b,1,h,w)
    patches = torch.nn.Unfold((3,3),padding=(1,1))(intensities).view(b,-1,h,w)
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + transf**2)    
    return transf_norm

def _hamming_distance(t1, t2):
    dist = (t1 - t2)**2
    dist_norm = dist / (0.1 + dist)
    dist_sum = torch.sum(dist_norm, 1, keepdims=True)
    return dist_sum

def ternary_loss(im1, im2_warped):

    t1 = _ternary_transform(im1)
    t2 = _ternary_transform(im2_warped)
    dist = _hamming_distance(t1, t2)

    return L1_Charbonnier_loss()(dist[:,:,1:-1,1:-1])

from torch import autograd
def calculate_gradient_penalty(real_images, fake_images, discrminator):
    eta = torch.FloatTensor(real_images.size(0),1,1,1).uniform_(0,1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3)).cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discrminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty    