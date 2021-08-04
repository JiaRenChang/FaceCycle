from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math

import RAFFace as DA
from models import *
import scipy.sparse
import scipy.sparse.linalg
import copy
from sklearn.metrics import confusion_matrix
from itertools import chain
from torchvision import models
import logging

parser = argparse.ArgumentParser(description='RAFFace')

parser.add_argument('--datapath', default='/media/jiaren/DataSet/basic/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= '/media/jiaren/DataSet/FaceCycle/ExpCode/ExpCode_19.tar',
                    help='load model')
parser.add_argument('--savemodel', default='/media/jiaren/DataSet/FaceCycle/ExpCode_test/RAFFace/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#SAVE
if not os.path.isdir(args.savemodel):
    os.makedirs(args.savemodel)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(args.savemodel+'modellog.log', 'a'))
print = logger.info

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageloader(args.datapath, train_fold='train'), 
         batch_size= 256, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageloader(args.datapath, train_fold='test'), 
         batch_size= 100, shuffle= False, num_workers= 4, drop_last=False)

class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.classify = nn.Sequential(nn.Linear(6272,7)) 
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0)         
                                   
    def forward(self, x):
        return self.classify(x)  
  
         
model = codegeneration().cuda()
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['codegeneration'])
model.eval()

loss_fc = nn.CrossEntropyLoss().cuda()

def train(img, label, optimizer, fc):

        if args.cuda:
            img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()

        fc.train()
        model.eval()

        exp_code  = model(img) 
        output = fc(exp_code.squeeze()) 

        loss = loss_fc(output, label)

        loss.backward()
        optimizer.step()

        return torch.mean(loss.data)

def test(img, label, fc):
        model.eval()
        fc.eval()        
        
        if args.cuda:
            img, label = img.cuda(), label.cuda()

        exp_code  = model(img)
        output = fc(exp_code.squeeze())  
         
        _, predicted = torch.max(output.data, 1)
        total = label.size(0)
        correct = predicted.eq(label.data).cpu()  
            
        return predicted.data.cpu(), label.data.cpu() , float(correct.sum())/float(total)

def adjust_learning_rate(optimizer, epoch):
    lr = 30.0 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

        fc = classifier().cuda()

        optimizer = optim.SGD([{"params":fc.parameters()}], lr=0.0001, momentum=0.9, weight_decay=0.0001, nesterov=False)
    
        start_full_time = time.time()
        for epoch in range(1, args.epochs+1):
            print('This is %d-th epoch' %(epoch))
            total_train_loss = 0
            adjust_learning_rate(optimizer,epoch)
            torch.cuda.empty_cache()

            ## training ##
            for batch_idx, (img, label) in enumerate(TrainImgLoader):  #imgL_crop, imgR_crop, disp_crop_L
                 start_time = time.time()
                 loss = train(img, label, optimizer, fc)
                 total_train_loss += loss
                 if batch_idx % 10 == 0:
                    print('epoch %d iter %d total training loss = %.3f' %(epoch, batch_idx, total_train_loss/batch_idx))

            savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
            if epoch % 50 ==0:
                torch.save({
                    'epoch': epoch,
                    'state_dict': fc.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                }, savefilename)
            
            total_test_loss = 0
            if epoch % 10 ==0:
                for batch_idx, (img, label) in enumerate(TestImgLoader):
                       preds, labels, loss = test(img, label, fc)
                       #print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
                       total_test_loss += loss
                print('total test accuracy = %.3f' %(100.0*total_test_loss/len(TestImgLoader)))

        #------------- TEST ------------------------------------------------------------
        total_test_loss = 0
        predm = []
        labelm = []
        torch.cuda.empty_cache()
        #val_log = open(args.savemodel+"/Validation_EPE.txt","a")
        for batch_idx, (img, label) in enumerate(TestImgLoader):
               preds, labels, loss = test(img, label, fc)
               #print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
               total_test_loss += loss
               predm.append(preds)
               labelm.append(labels)
        print('total test accuracy = %.3f' %(100.0*total_test_loss/len(TestImgLoader)))
        #----------------------------------------------------------------------------------

if __name__ == '__main__':
   main()
    
