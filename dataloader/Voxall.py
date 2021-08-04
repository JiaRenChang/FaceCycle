import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from . import preprocess 
import random
from itertools import chain
import time
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
	tries = 2
	for i in range(tries):
		try:
			img = Image.open(path).convert('RGB')
		except OSError as e:
			if i < tries - 1: # i is zero indexed
				continue
			else:
				print(path)
				return None
	return img


class myImageloader(data.Dataset):
    def __init__(self, datapath='./dataloader/Vox1.txt', loader=default_loader):

        self.alldatalist = []
        fp = open (datapath,"r")
        for line in fp.readlines():
            line = line.strip()
            #print(line)
            self.alldatalist.append(line)
        fp.close()
        #print(len(self.alldatalist))

        self.loader = loader

    def __getitem__(self, index):

        identity_dir = self.alldatalist[index] + '/'
        split_path = identity_dir.split('/')       
        id_img_list = [identity_dir+'/'+img.name for img in os.scandir(identity_dir) if is_image_file(img.name)]

        #random select one image
        img_idx = np.random.randint(0, len(id_img_list)-1)
        img_idx2 = np.random.randint(0, len(id_img_list)-1)        

        img1 = self.loader(id_img_list[img_idx])

        while img1 is None:
            img_idx = np.random.randint(0, len(id_img_list)-1)
            img1 = self.loader(id_img_list[img_idx])

        processed1 = preprocess.get_transform(augment=True)  
        img1 = processed1(img1)
                      
        if np.random.rand() > 0.5:
            img2 = transforms.functional.hflip(img1) 
        else:
            img2 = self.loader(id_img_list[img_idx2])
            while img2 is None:
                img_idx2 = np.random.randint(0, len(id_img_list)-1)      
                img2 = self.loader(id_img_list[img_idx2])  
            processed2 = preprocess.get_transform(augment=True)              
            img2 = processed2(img2)
             
        return img1, img2


    def __len__(self):
        return len(self.alldatalist)
