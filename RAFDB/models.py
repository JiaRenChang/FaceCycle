import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from submodule import *

class codegeneration(torch.nn.Module):
    def __init__(self):
        super(codegeneration, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,7,2,3, bias=True),                         
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64,64,3,1,1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64,64,3,1,1, bias=True),                                  
                                   nn.LeakyReLU(negative_slope=0.1),
                                   selfattention(64),
                                   nn.Conv2d(64,64,3,1,1, bias=True),                                                             
                                   nn.LeakyReLU(negative_slope=0.1)) #64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64,128,3,2,1, bias=True),                             
                                   nn.LeakyReLU(negative_slope=0.1),
                                   selfattention(128),
                                   nn.Conv2d(128,128,3,1,1, bias=True),                                                            
                                   nn.LeakyReLU(negative_slope=0.1),) #64

        self.resblock1 = BasicBlockNormal(128,128)
        self.resblock2 = BasicBlockNormal(128,128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128,128,3,2,1, bias=True), 
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(128,128,3,1,1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),) #64                       

        self.layer3_1 = nn.Sequential(nn.Conv2d(128,256,3,2,1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(256,256,3,1,1, bias=True), 
                                   nn.LeakyReLU(negative_slope=0.1),) #64

        self.layer3_2 = nn.Sequential(nn.Conv2d(256,256,3,1,1, bias=True),  # stride 2 for 128x128
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(256,128,3,1,1, bias=True),                                                            
                                   nn.LeakyReLU(negative_slope=0.1)) #64
                       
        self.expresscode = nn.Sequential(nn.Linear(2048,512),
                                       nn.LeakyReLU(negative_slope=0.1), 
                                       nn.Linear(512,256)) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #encoder
        out_1 = self.conv1(x) 
        out_1 = self.layer1(out_1)
        out_2 = self.layer2_1(out_1)
        out_2 = self.resblock1(out_2)
        out_2 = self.resblock2(out_2)
        out_2 = self.layer2_2(out_2)
        out_3 = self.layer3_1(out_2)
        out_3 = self.layer3_2(out_3)
        out_3 = out_3.view(x.size()[0],-1)
        # we use out_3 features for RAF-DB expression recognition    
        return out_3
 