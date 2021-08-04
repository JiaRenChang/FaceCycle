import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import preprocess 
import random
from itertools import chain

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageloader(data.Dataset):
    def __init__(self, datapath='/media/jiaren/DataSet/basic/', train_fold = 'train', loader=default_loader):
 
        imgs_list = open(os.path.join(datapath,'EmoLabel/list_patition_label.txt'))

        self.imglist = []
        self.labels = []
        for image in imgs_list:            
            image = image.split(' ')
            if train_fold in image[0]:
                self.imglist.append(os.path.join(datapath,'Image/aligned/',image[0][:-4]+'_aligned.jpg'))
                self.labels.append(int(image[1][0])-1)

        self.loader = loader

        if train_fold == 'train':
            self.training = True
        else:
            self.training = False
                    
    def __getitem__(self, index):
        if self.training:
            processed = preprocess.get_transformtensor(augment=True) 
        else:
            processed = preprocess.get_transformtensor(augment=False) 
            
        img =  self.loader(self.imglist[index])
        img =  processed(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imglist) #12271 #
        
if __name__ == '__main__':
    a = myImageloader()

