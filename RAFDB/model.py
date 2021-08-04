import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from .submodule import *

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
        expcode = self.expresscode(out_3)
        expcode = expcode.view(x.size()[0],-1,1,1)   
        expcode = F.normalize(expcode, p=2, dim=1)
        return expcode
 
        
class exptoflow(torch.nn.Module):
    def __init__(self):
        super(exptoflow, self).__init__()    
            
        self.motiongen1 = nn.Sequential(nn.Conv2d(256,4096,1,1,0),
                                        nn.PixelShuffle(4),
                                        nn.LeakyReLU(negative_slope=0.1),
                                        selfattention(256),
                                        nn.Conv2d(256,256,3,1,1),
                                        nn.LeakyReLU(negative_slope=0.1),) #to 4*4
                                                                                                                                                                                               
        self.motiongen2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(256,128,3,1,1), 
                                        nn.LeakyReLU(negative_slope=0.1),
                                        selfattention(128),
                                        nn.Conv2d(128,128,3,1,1),
                                        nn.LeakyReLU(negative_slope=0.1),) #to 8*8 

        self.resblock1 = BasicBlockNormal(128,128)
        self.resblock2 = BasicBlockNormal(128,128)  

        self.toflow4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    nn.Conv2d(64, 2,1,1,0, bias=False)) #16x16

        self.normact = nn.Tanh()                           

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.toflow4:
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.05)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        Bf,Cf,Hf,Wf = flo.size()        
        ## down sample flow
        #scale = H/Hf
        #flo = F.upsample(flo, size = (H,W),mode='bilinear', align_corners=True)  # resize flow to x  
        #flo = torch.clamp(flo,-1,1)        #       
        #flo = flo*scale # rescale flow depend on its size
        ##
        
        # mesh grid 
        xs = np.linspace(-1,1,W)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1,1,1).cuda()

        vgrid = Variable(xs, requires_grad=False) + flo.permute(0,2,3,1)         
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        output = torch.clamp(output,-1,1)
        return output 
        
    def forward(self, expcode):
        
        motion = self.motiongen1(expcode)
        motion = self.motiongen2(motion) 
        motion = self.resblock1(motion)
        motion = self.resblock2(motion)
        flow = self.normact(self.toflow4(motion))
        backflow = self.warp(flow.clone(), flow)*-1.0 
        
        return flow, backflow 
        
from torchvision import models

class generator(torch.nn.Module):
    def __init__(self, is_exp = False):
        super(generator, self).__init__()   
        self.is_exp = is_exp
        vgg_pretrained_cnn = models.vgg19(pretrained=True).features
        #self.vgg_pretrained_classifer = models.vgg19(pretrained=False).classifier

        self.conv1_1 = nn.Sequential(vgg_pretrained_cnn[0],vgg_pretrained_cnn[1])
        self.conv1_2 = nn.Sequential(vgg_pretrained_cnn[2],vgg_pretrained_cnn[3])

        self.conv2_1 = nn.Sequential(vgg_pretrained_cnn[5],vgg_pretrained_cnn[6])
        self.conv2_2 = nn.Sequential(vgg_pretrained_cnn[7],vgg_pretrained_cnn[8])

        self.conv3_1 = nn.Sequential(vgg_pretrained_cnn[10],vgg_pretrained_cnn[11])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               m.weight.requires_grad = False
               m.bias.requires_grad = False

        self.inplanes = 256
        self.redconv = nn.Sequential(nn.Conv2d(256, 256, 3,1,1,1), nn.LeakyReLU(negative_slope=0.1))

        self.cnn = self._make_layer(BasicBlockNormal, 256, 8, stride=1) #2

        self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.LeakyReLU(negative_slope=0.1),
                                nn.Conv2d(128,128,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1))                               
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.LeakyReLU(negative_slope=0.1),
                                nn.Conv2d(128,128,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1))
        
        self.torgb = nn.Conv2d(128, 3, 3, 1, 1)
          
        self.noise_encoding1 = nn.Sequential(nn.Conv2d(128,128,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1),
                                nn.Conv2d(128,128,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1))
        self.noise_encoding0 = nn.Sequential(nn.Conv2d(256,256,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1),
                                nn.Conv2d(256,256,3,1,1,bias=True),
                                nn.LeakyReLU(negative_slope=0.1))
        

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes * block.expansion, 1,stride,0),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        Bf,Cf,Hf,Wf = flo.size() 

        ## down sample flow
        #scale = H/Hf
        flo = F.upsample(flo, size = (H,W),mode='bilinear', align_corners = True)  # resize flow to x
        #occmap = F.upsample(occmap, size = (H,W),mode='bilinear')
        #flo = flo*scale # rescale flow depend on its size
        ##
        
        # mesh grid 
        xs = np.linspace(-1,1,W)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1,1,1).cuda()

        vgrid = Variable(xs, requires_grad=False) + flo.permute(0,2,3,1)         
        output = nn.functional.grid_sample(x, vgrid, align_corners = True)
        
        return output

    @staticmethod
    def denorm(x):
        x = x.clone()
        x[:,0,:,:] = (x[:,0,:,:]- 0.485) / 0.229 
        x[:,1,:,:] = (x[:,1,:,:]- 0.456) / 0.224 
        x[:,2,:,:] = (x[:,2,:,:]- 0.406) / 0.225 
        return x

    def forward(self, x , flow=None):
        #x = self.denorm(x)
        #with torch.no_grad():

        feat = self.conv1_1(x)
        feat = self.conv1_2(feat)
        feat = F.max_pool2d(feat,kernel_size=2, stride=2, padding=0, dilation=1)
        feat2 = self.conv2_1(feat)
        feat = self.conv2_2(feat2)
        feat = F.max_pool2d(feat,kernel_size=2, stride=2, padding=0, dilation=1)
        feat = self.conv3_1(feat)

        global_face = F.adaptive_avg_pool2d(feat,1)
        global_face1 = F.adaptive_avg_pool2d(feat2,1)
        batch, _, height, width = feat.shape
        noise = feat.new_empty(batch, 256, height, width).normal_()
        face_res0 = self.noise_encoding0(noise+global_face)
        batch2, _, height2, width2 = feat2.shape
        noise2 = feat.new_empty(batch2, 128, height2, width2).normal_()
        face_res1 = self.noise_encoding1(noise2+global_face1)

        if flow is not None:
            deform_feat = self.warp(feat, flow)  + face_res0
            deform_feat2 = self.warp(feat2, flow) + face_res1         
        else:
            deform_feat = feat + face_res0
            deform_feat2 = feat2 + face_res1

        deform_feat = self.redconv(deform_feat)
        out0 = self.cnn(deform_feat)
        out = self.up(out0)
        out1 = self.up2(torch.cat([deform_feat2,out],dim=1))
        face = self.torgb(out1)

        return  face

class normliztor(torch.nn.Module):
    def __init__(self, is_exp = False):
        super(normliztor, self).__init__()   
        self.is_exp = is_exp
        vgg_pretrained_cnn = models.vgg19(pretrained=True).features
        #self.vgg_pretrained_classifer = models.vgg19(pretrained=False).classifier

        self.conv1_1 = nn.Sequential(vgg_pretrained_cnn[0],vgg_pretrained_cnn[1])
        self.conv1_2 = nn.Sequential(vgg_pretrained_cnn[2],vgg_pretrained_cnn[3])

        self.conv2_1 = nn.Sequential(vgg_pretrained_cnn[5],vgg_pretrained_cnn[6])
        self.conv2_2 = nn.Sequential(vgg_pretrained_cnn[7],vgg_pretrained_cnn[8])

        self.conv3_1 = nn.Sequential(vgg_pretrained_cnn[10],vgg_pretrained_cnn[11])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               m.weight.requires_grad = False
               m.bias.requires_grad = False

        self.inplanes = 256
        self.denorm = SPADE(256,256)
        self.renorm = SPADE(256,256)

        self.cnn = self._make_layer(BasicBlockNormal, 256, 6, stride=1) #2
        self.cnn3 = self._make_layer(BasicBlockNormal, 256, 1, stride=1) #2

        self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(128,128,3,1,1,bias=True),
                                nn.ReLU(),                                
                                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
                                nn.ReLU())
        
        self.torgb = nn.Conv2d(128, 3, 1, 1, 0)
          

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes * block.expansion, 1,stride,0),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  

    def forward(self, x, denorm=True, code=None):
        #x = self.denorm(x)
        #with torch.no_grad():

        feat = self.conv1_1(x)
        feat = self.conv1_2(feat)
        feat = F.max_pool2d(feat,kernel_size=2, stride=2, padding=0, dilation=1)
        feat = self.conv2_1(feat)
        feat = self.conv2_2(feat)
        feat = F.max_pool2d(feat,kernel_size=2, stride=2, padding=0, dilation=1)
        feat = self.conv3_1(feat)

        if denorm:
            deform_feat = self.denorm(feat, code)
        else:
            deform_feat = self.renorm(feat, code)

        out = self.cnn(deform_feat)
        out = self.cnn3(out)  
        out = self.up(out)
        face = self.torgb(out)
        
        return  face