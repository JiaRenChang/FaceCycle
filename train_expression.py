import time
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from  vgg19 import *
import random
from dataloader import Voxall as DA
from models import *
from Loss import *

parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--datapath', default='./dataloader/Vox1.txt',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= '/media/jiaren/DataSet/FaceCycle/ExpCode/ExpCode_4.tar',
                    help='load model')
parser.add_argument('--savemodel', default='/media/jiaren/DataSet/FaceCycle/ExpCode/',
                    help='save model')                    
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
torch.manual_seed(2)
torch.cuda.manual_seed(4)

save_image_fold = args.savemodel + 'imgs/'
if not os.path.isdir(save_image_fold):
    os.makedirs(save_image_fold)

def fast_collate(batch):
    imgs0 = [img[0] for img in batch]
    imgs1 = [img[1] for img in batch]

    w = imgs0[0].size[0]
    h = imgs0[0].size[1]

    tensor0 = torch.zeros( (len(imgs0), 3, h, w), dtype=torch.uint8)
    tensor1 = torch.zeros( (len(imgs1), 3, h, w), dtype=torch.uint8)

    for i, img in enumerate(imgs0):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor0[i] += torch.from_numpy(nump_array)

    for i, img in enumerate(imgs1):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor1[i] += torch.from_numpy(nump_array)

    return tensor0, tensor1

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input0, self.next_input1 = next(self.loader)
        except StopIteration:
            self.next_input0, self.next_input1 = None, None
            return

        with torch.cuda.stream(self.stream):
            self.next_input0 = self.next_input0.cuda(non_blocking=True)
            self.next_input1 = self.next_input1.cuda(non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input0 = self.next_input0.float()
            self.next_input0 = self.next_input0.sub_(self.mean).div_(self.std)
            self.next_input1 = self.next_input1.float()
            self.next_input1 = self.next_input1.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input0 = self.next_input0
        input1 = self.next_input1
        self.preload()
        return input0, input1


TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageloader(), 
         batch_size= 16, shuffle= True, pin_memory=True, collate_fn=fast_collate, num_workers= 12, drop_last=True)


def denorm(x):
	x[:,0,:,:] = x[:,0,:,:]*0.229 + 0.485
	x[:,1,:,:] = x[:,1,:,:]*0.224 + 0.456
	x[:,2,:,:] = x[:,2,:,:]*0.225 + 0.406
	return x	
	
def denorm_reto(x):
    y = x.clone()
    y[:,0,:,:] = ((x[:,2,:,:]*0.229 + 0.485)*255-91.4953)
    y[:,1,:,:] = ((x[:,1,:,:]*0.224 + 0.456)*255-103.8827)
    y[:,2,:,:] = ((x[:,0,:,:]*0.225 + 0.406)*255-131.0912)
    return y.contiguous()	

device = torch.device('cuda')
vgg = Vgg19(requires_grad=False)

if torch.cuda.is_available():
	vgg = nn.DataParallel(vgg)
	vgg.to(device)
	vgg.eval()
feat_layers = ['r21','r31','r41']

codegeneration =  codegeneration().cuda()
exptoflow =  exptoflow().cuda()
Swap_Generator = generator().cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    Swap_Generator.load_state_dict(state_dict['Swap_Generator'])
    codegeneration.load_state_dict(state_dict['codegeneration'])
    exptoflow.load_state_dict(state_dict['exptoflow'])

#codegeneration =  nn.DataParallel(codegeneration)
#exptoflow =  nn.DataParallel(exptoflow)
#Swap_Generator = nn.DataParallel(Swap_Generator)

optimizer =  optim.Adam([{"params":Swap_Generator.parameters()},{"params":codegeneration.parameters()},{"params":exptoflow.parameters()}], lr=1e-5, betas=(0.5, 0.999))
pytorch_total_params = sum(p.numel() for p in codegeneration.parameters())

def adjust_learning_rate(optimizer, epoch):
    lr = 5e-5
    if epoch >= 10 and epoch < 20:
        lr = 5e-5 
    elif epoch >= 20 and epoch < 30:
        lr = 1e-5
    elif epoch >= 30 and epoch < 40:
        lr = 1e-5
    elif epoch >= 40:
        lr = 5e-6 
    elif epoch == 0:
        lr = 1e-5
             
    print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warp(x, flo):
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
    flo = torch.clamp(flo,-1,1)
    #flo = flo*scale # rescale flow depend on its size
    ##    
    # mesh grid 
    xs = np.linspace(-1,1,W)
    xs = np.meshgrid(xs, xs)
    xs = np.stack(xs, 2)
    xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1,1,1).to(device)

    vgrid = Variable(xs, requires_grad=False) + flo.permute(0,2,3,1)         
    output = nn.functional.grid_sample(x, vgrid, align_corners = True)
    
    return output

def forwardloss(im_id0, im_id1, batch):
    #with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        b,c,h,w = im_id0.size()
        # actually runs in batch 32
        full_batch = torch.cat([im_id0, im_id1],dim=0)
            
        expcode = codegeneration(full_batch.data)
        flow_full, backflow_full = exptoflow(expcode)
       
        # generate neutral faces
        neu_face = Swap_Generator(full_batch.data, flow_full)

        # swap face
        split_face = torch.split(neu_face, [b,b], dim=0)
        swap_face = torch.cat([split_face[1], split_face[0]], dim=0)

        # recontruct original faces by direct warp neutral
        direct_forward = warp(full_batch, flow_full)
        split_direct_face = torch.split(direct_forward, [b,b], dim=0)
        swap_direct_face = torch.cat([split_direct_face[1], split_direct_face[0]], dim=0)
        direct_backward = warp(swap_direct_face, backflow_full)
        swap_backward = warp(swap_face, backflow_full)

        # recontruct original faces by warp neutral face features
        recon_face = Swap_Generator(swap_face, backflow_full) #detach or not

        # generator need to recontruct original faces without flow
        rec_img = Swap_Generator(full_batch.data, None)

        #photometric loss        
        pixel_loss = F.l1_loss(recon_face, full_batch.data) + \
                     F.l1_loss(rec_img, full_batch.data) + \
                     F.l1_loss(swap_backward, full_batch.data) + \
                     1.5*F.l1_loss(direct_backward, full_batch.data)  
                
        #percetual loss
        perc_full = torch.cat([full_batch, full_batch, full_batch, full_batch],dim=0) #
        perc_full = perc_full.clone()
        rec_full = torch.cat([neu_face, swap_backward, direct_backward, recon_face],dim=0) #
        im_feat = vgg(perc_full, feat_layers)
        rec_feat = vgg(rec_full, feat_layers)

        pec_loss0 = perceptual_loss(rec_feat[0],im_feat[0]) 
        pec_loss1 = perceptual_loss(rec_feat[1],im_feat[1])
        pec_loss2 = perceptual_loss(rec_feat[2],im_feat[2])

        perc_loss = pec_loss0 + pec_loss1 + pec_loss2

        # SSIM loss
        s_loss = ssim_loss(full_batch, recon_face) + \
                 ssim_loss(full_batch, rec_img) + \
                 1.5*ssim_loss(full_batch, direct_backward)

        # neutral face symetric loss
        sym_loss = symetricloss(neu_face) + 0.5*symetricloss(direct_forward)
        
        loss = 0.01*perc_loss + pixel_loss + 1.5*sym_loss + 5.0*s_loss
        loss.backward()

        optimizer.step()
                                                      
        if batch % 100 == 0:
            print('iter %d percetual loss: %.2f pixel_loss: %.2f ssim loss: %.2f symetric: %.2f' %(batch, perc_loss, pixel_loss, s_loss, sym_loss))

        if batch % 1000 == 0:
            save_image(torch.cat((denorm(full_batch[0:8].data)\
                    , denorm(direct_backward[0:8].data) \
                    , denorm(recon_face[0:8].data)
                    , denorm(neu_face[0:8].data)\
                    ),0), os.path.join(save_image_fold, '{}_{}_decode.png'.format(epoch,int(batch/1000))))


if __name__ == '__main__':
    for epoch in range(0,40):
        adjust_learning_rate(optimizer, epoch)
        prefetcher = data_prefetcher(TrainImgLoader)

        input0, input1 = prefetcher.next()
        
        batch_idx = 0
        while input0 is not None:    
            input0, input1 = input0.cuda(), input1.cuda()

            codegeneration.train()
            exptoflow.train()
            Swap_Generator.train()
            forwardloss(input0, input1, batch_idx) 
            batch_idx += 1 
            input0, input1 = prefetcher.next()
              
        #SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        # model.module.state_dict() for nn.dataparallel    
        savefilename = args.savemodel + 'ExpCode_'+str(epoch)+'.tar'
        torch.save({'codegeneration':codegeneration.state_dict(),
                    'exptoflow':exptoflow.state_dict(),
                    'Swap_Generator':Swap_Generator.state_dict(),              
        }, savefilename)
		



