import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

class selfattention(nn.Module):
    def __init__(self, inplanes):
        super(selfattention, self).__init__()

        self.interchannel = inplanes
        self.inplane = inplanes
        self.g     = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.phi   = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        b,c,h,w = x.size()
        g_y = self.g(x).view(b, c, -1) #BXcXN        
        theta_x = self.theta(x).view(b, self.interchannel, -1) 
        theta_x = F.softmax(theta_x, dim = -1) # softmax on N       
        theta_x = theta_x.permute(0,2,1).contiguous() #BXNXC'
        
        phi_x = self.phi(x).view(b, self.interchannel, -1) #BXC'XN
       
        similarity = torch.bmm(phi_x, theta_x) #BXc'Xc'

        g_y = F.softmax(g_y, dim = 1)
        attention = torch.bmm(similarity, g_y) #BXCXN
        attention = attention.view(b,c,h,w).contiguous()
        y = self.act(x + attention)
        return y

class BasicBlockNormal(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNormal, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes,planes,3,stride,1)
        self.relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = (out + identity)
        return self.relu(out)


class SPADE(torch.nn.Module):
    def __init__(self, in_plane, out_plane):
        super(SPADE, self).__init__()

        self.bn = nn.InstanceNorm2d(out_plane, affine=False)
                                                                      				              
        self.beta  = nn.Conv2d(in_plane,out_plane,1,1,0)
        self.gamma = nn.Conv2d(in_plane,out_plane,1,1,0)

    def forward(self, x, code):
        beta = self.beta(code)
        gamma = self.gamma(code)
        x = self.bn(x)*gamma + beta
        return x