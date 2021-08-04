import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageFilter

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
                   


class RandomBlur(object):
    def __init__(self, p=0.5, kernel_size=[0.1,2.0]):
        self.kernel_size = kernel_size
        self.p = p
    def __call__(self, img):
        radius = random.uniform(self.kernel_size[0], self.kernel_size[1])
        if random.random() > self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img

def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
	transforms.Resize(64),
    transforms.RandomResizedCrop(64, scale=(0.95, 1.05), ratio=(0.95, 1.05)),
	transforms.CenterCrop(64),
	transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    #transforms.Normalize(**normalize),
    ]
    
    return transforms.Compose(t_list)


def test_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
	    transforms.Resize(64),
	    transforms.CenterCrop(64),
        #transforms.Grayscale(num_output_channels=1),
        #transforms.ToTensor(),
        #transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)

def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 64
    if augment:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    else:
            return test_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)


def scale_crop_t(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
	transforms.Resize(100),
    transforms.RandomResizedCrop(100, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
	transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def test_crop_t(input_size=64, scale_size=None, normalize=__imagenet_stats):
    t_list = [
	    transforms.Resize(100),
	    transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
        ]
    return transforms.Compose(t_list)




def get_transformtensor(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 64
    if augment:
            return scale_crop_t(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    else:
            return test_crop_t(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)