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
import argparse
from models import *
from dataloader import preprocess as preprocess
from OpticalFlow_Visualization import flow_vis


parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--loadmodel', default= './finalmodel.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./Test_translation/',
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

save_image_fold = args.savemodel
if not os.path.isdir(save_image_fold):
    os.makedirs(save_image_fold)

def denorm(x):
	x[:,0,:,:] = x[:,0,:,:]*0.229 + 0.485
	x[:,1,:,:] = x[:,1,:,:]*0.224 + 0.456
	x[:,2,:,:] = x[:,2,:,:]*0.225 + 0.406
	return x.clamp(0,1)
	
def denorm_reto(x):
	x[:,0,:,:] = ((x[:,0,:,:]*0.229 + 0.485)-0.5)*0.5
	x[:,1,:,:] = ((x[:,1,:,:]*0.224 + 0.456)-0.5)*0.5
	x[:,2,:,:] = ((x[:,2,:,:]*0.225 + 0.406)-0.5)*0.5
	return x.clamp(0,1)	

device = torch.device('cuda')


idcodegen = codegeneration().cuda()
Swap_Norm = normalizer().cuda()

codegeneration = codegeneration().cuda()
exptoflow = exptoflow().cuda()
Swap_Generator = generator().cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    codegeneration.load_state_dict(state_dict['codegeneration'])
    exptoflow.load_state_dict(state_dict['exptoflow'])
    Swap_Generator.load_state_dict(state_dict['Swap_Generator'])
    idcodegen.load_state_dict(state_dict['idcodegen'])
    Swap_Norm.load_state_dict(state_dict['Swap_Norm'])


def forwardloss(im_id1, im_id2, idx):

        #with torch.no_grad():
        expcode1  = codegeneration(im_id1)   ## predict  expression -> neutral
        expcode2  = codegeneration(im_id2)   ## predict  expression -> neutral        

        flow1, backflow1 = exptoflow(expcode1)                               
        flow2, backflow2 = exptoflow(expcode2)            
            
        neu_face1 = Swap_Generator(im_id1,flow1)
        neu_face2 = Swap_Generator(im_id2,flow2)

        rec_face1 = Swap_Generator(neu_face1, backflow2)        
        rec_face2 = Swap_Generator(neu_face2, backflow1) 
                
        # general identity code
        id_code_1  = idcodegen(im_id1)   ## predict  expression -> neutral 
        id_code_2  = idcodegen(im_id2)   ## predict  expression -> neutral 

        global_mean_face_1 = Swap_Norm(neu_face1, True, id_code_1)
        global_mean_face_2 = Swap_Norm(neu_face2, True, id_code_2)

        neu_face1r = Swap_Norm(global_mean_face_2, False, id_code_1)
        neu_face2r = Swap_Norm(global_mean_face_1, False, id_code_2)        
    
        id1_rec_face = Swap_Generator(neu_face1r, backflow2)
        id2_rec_face = Swap_Generator(neu_face2r, backflow1)
                

        flow1 = F.upsample(flow1, size = (64,64),mode='nearest').clamp(-1,1)
        flow2 = F.upsample(flow2, size = (64,64),mode='nearest').clamp(-1,1)

        flow_color = flow_vis.flow_to_color(flow1[0].data.squeeze().cpu().permute(1,2,0).numpy(), convert_to_bgr=False)                       
        #flow_color_inv = flow_vis.flow_to_color(backflow1[0].data.squeeze().cpu().permute(1,2,0).numpy(), convert_to_bgr=False) 
                              
        flow_color2 = flow_vis.flow_to_color(flow2[0].data.squeeze().cpu().permute(1,2,0).numpy(), convert_to_bgr=False)                       
        #flow_color_inv1 = flow_vis.flow_to_color((id1_backflow).data.squeeze().cpu().permute(1,2,0).numpy(), convert_to_bgr=False)
                
        save_image(torch.cat((denorm(im_id1.data.clone()), denorm(im_id2.data.clone()), \
                            denorm(neu_face1.data.clone()), denorm(neu_face2.data)  \
                            , denorm(rec_face1.data) \
                            , denorm(rec_face2.data) \
                            , denorm(global_mean_face_1.data) \
                            , denorm(global_mean_face_2.data) \
                            , denorm(neu_face1r.data) \
                            , denorm(neu_face2r.data) \
                            ,transforms.ToTensor()(flow_color).cuda().unsqueeze(0) \
                            ,transforms.ToTensor()(flow_color2).cuda().unsqueeze(0) \
                            ),0), os.path.join(save_image_fold, 'translation'+ str(idx) + '.png'))
                            
processed = preprocess.test_crop_t()  

if __name__ == '__main__':

    codegeneration.eval()
    exptoflow.eval()
    Swap_Generator.eval()
    Swap_Norm.eval()
    idcodegen.eval()
    
    driver_imgs =['./Imgs/id1.jpg']
    source_imgs =['./Imgs/id2.jpg']

    im_id1 = processed(Image.open(source_imgs[0]).convert('RGB')).unsqueeze(0).to(device)

    for i in range(len(driver_imgs)):
        im_id0 = processed(Image.open(driver_imgs[i]).convert('RGB')).unsqueeze(0).to(device)
        forwardloss(im_id0, im_id1, i) 



