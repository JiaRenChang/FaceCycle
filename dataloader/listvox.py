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

datapath='/media/jiaren/DataSet/dense-face-frames/unzippedIntervalFaces/data/'

identitydirlist = os.listdir(datapath)
youtulist = []
for iddir in identitydirlist:
    youtulist.append([datapath+iddir+'/1.6/'+youtudir for youtudir in os.listdir(datapath+iddir+'/1.6/')])
youtulist = list(chain(*youtulist)) 
   
alldatalist = []
for youtudir in youtulist:
        alldatalist.append([youtudir+'/'+datadir for datadir in os.listdir(youtudir)])         
alldatalist = list(chain(*alldatalist)) 

'''
vox2path = '/media/jiaren/DataSet/vox2/mp4/'
voxall = []
for voxids in os.listdir(vox2path):
    voxall.append([vox2path+voxids+'/'+youtudir for youtudir in os.listdir(vox2path+voxids)])
voxall = list(chain(*voxall))

alldatalist += voxall
'''

print(len(alldatalist))  
aalldataaa = []  
for ytddir in alldatalist:
    if len([img for img in os.listdir(ytddir) if is_image_file(img)]) > 3:
        aalldataaa.append(ytddir)

print(len(aalldataaa)) 

with open ("Vox1.txt","w")as fp:
   for line in aalldataaa:
       fp.write(line+"\n")