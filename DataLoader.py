
# coding=utf-8 

import sys

import torchvision
import torch.nn as nn
from os.path import exists, join
from torchvision.transforms import Lambda,Compose, CenterCrop, Grayscale,RandomRotation,ToPILImage,RandomAffine,ToTensor,ColorJitter, RandomVerticalFlip,Normalize, Scale,RandomCrop,Pad,RandomHorizontalFlip,RandomCrop,CenterCrop

import torchvision.transforms.functional as TF

from torchvision.transforms.functional import rotate
import numpy.random as random
from torchvision.datasets.folder import IMG_EXTENSIONS
import matplotlib.pyplot as plt
import torch.utils.data as data
from os import listdir
import nibabel as nib
from skimage.transform import rescale, resize, downscale_local_mean

import os
import PIL
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import collections
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
import cv2
from scipy import ndimage
from skimage.filters import threshold_otsu


import argparse


from skimage.transform import resize

def get_data(opt,dataPath,DataloaderType='training',Module='Encoder'): #validation, 'test'
   
    if opt.Dataset=='DRIVE' or opt.Dataset=='STARE'or opt.Dataset=='IOSTAR':
        return Read_DRIVE_Dataset(opt,join(dataPath, DataloaderType),Module)
    else:
        return ReadTheDataset(opt,join(dataPath, DataloaderType),Module)
    
    
def CropCentre(image,segmentation,ImgSize):
    image=np.float32(image)
    segmentation=np.float32(segmentation)

    #print(image.shape,segmentation.shape)
    
    w=image.shape[0]
    h=image.shape[1]
    #print(w,h)
   
    if w>ImgSize and h>ImgSize:
        i=int((w-ImgSize)/2.)
        j=int((h-ImgSize)/2.)


        if image.ndim==3:
            imageNew=image[i:i+ImgSize, j:j+ImgSize,:]
            segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize,:]
        elif image.ndim==2:

            imageNew=image[i:i+ImgSize, j:j+ImgSize]
            segmentationNew=segmentation[i:i+ImgSize, j:j+ImgSize]

    elif h<ImgSize and w>ImgSize:
        i=int((w-ImgSize)/2.)
        j=int((-h+ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[:,j:j+h,:]=image[i:i+ImgSize, :,:]
            segmentationNew[:,j:j+h,:]=segmentation[i:i+ImgSize, :,:]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[:,j:j+h]=image[i:i+ImgSize, :]
            segmentationNew[:,j:j+h]=segmentation[i:i+ImgSize, :]
        

            
    elif h>ImgSize and w<ImgSize:
        i=int((-w+ImgSize)/2.)
        j=int((+h-ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[i:i+w,:,:]=image[:,j:j+ImgSize, :]
            segmentationNew[i:i+w,:,:]=segmentation[:,j:j+ImgSize, :]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[i:i+w,:]=image[:,j:j+ImgSize]
            segmentationNew[i:i+w,:]=segmentation[:,j:j+ImgSize]
        
           
    elif h<ImgSize and w<ImgSize:
        i=int((-w+ImgSize)/2.)
        j=int((-h+ImgSize)/2.)


        if image.ndim==3:
            imageNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))
            segmentationNew=np.zeros((ImgSize,ImgSize,image.shape[-1]))

            imageNew[i:i+w,j:j+h,:]=image#[:,:, :]
            segmentationNew[i:i+w,j:j+h,:]=segmentation#[i:i+ImgSize,j:j+ImgSize, :]
        elif image.ndim==2:
            imageNew=np.zeros((ImgSize,ImgSize))
            #pdb.set_trace()
            segmentationNew=np.zeros((ImgSize,ImgSize))
            
            imageNew[i:i+w,j:j+h]=image#[i:i+ImgSize,j:j+ImgSize]
            segmentationNew[i:i+w,j:j+h]=segmentation#[i:i+ImgSize,j:j+ImgSize]
            
    elif h==ImgSize and w==ImgSize:
        
            imageNew=image#[i:i+ImgSize,j:j+ImgSize]
            segmentationNew=segmentation#[i:i+ImgSize,j:j+ImgSize]

            
            
    return imageNew,segmentationNew

def ImageTransforms(image, segmentation,ImgSize=128, training=True, Module='Encoder'):
    
    #print(np.max(np.array(image)),np.max(np.array(segmentation)))
    
    image=Image.fromarray((255*np.array(image)).astype(np.uint8)) 
    segmentation=Image.fromarray(255*np.array(segmentation))

    
    if training:
        if Module=='Encoder':
        
        
            aa=np.random.randint(-30,10,1)
            if aa>0:
                scale=np.random.randint(101,120,1)*0.01
            elif aa<-15:
                scale=np.random.randint(80,99,1)*0.01
            else:
                scale= 1                

            aa=np.random.randint(-20,20,1)
            
            '''
            if aa>10:
                image=TF.hflip(image)
                segmentation=TF.hflip(segmentation)        
            '''
            #scale=np.random.randint(80,120,1)*0.01

            aa=np.random.randint(-20,20,1)
            shear=0
            if aa>10:
                shear=(np.random.randint(-10,10,1),np.random.randint(-10,10,1))


            #angle=45*(2*np.random.rand(1)[0]-1)
            angle=0
            translate=[np.random.randint(-30,30,1),np.random.randint(-30,30,1)]



            image=TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)     
            segmentation=TF.affine(segmentation, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)




            aa=np.random.randint(-100,100,5)
            Threshold=0         
            if aa[0]<Threshold:
                gamma=np.random.randint(60,150,1)[0]*0.01
                image=TF.adjust_gamma(image, gamma, gain=1)

            if aa[2]<Threshold:
                saturation_factor=np.random.randint(60,150,1)*0.01
                image=TF.adjust_saturation(image, saturation_factor)
            if aa[3]<Threshold:
                contrast_factor=np.random.randint(80,120,1)*0.01
                image=TF.adjust_contrast(image, contrast_factor)              



            #contrast_factor=np.random.randint(80,120,1)*0.01
            #image=TF.adjust_contrast(image, contrast_factor) 
        
        
        else: # Training=='Original':

            aa=np.random.randint(-30,10,1)
            if aa>0:
                scale=np.random.randint(101,120,1)*0.01
            elif aa<-15:
                scale=np.random.randint(80,99,1)*0.01
            else:
                scale= 1                



            #scale=np.random.randint(80,120,1)*0.01

            aa=np.random.randint(-20,20,1)
            shear=0
            if aa>10:
                shear=(np.random.randint(-10,10,1),np.random.randint(-10,10,1))



            translate=[np.random.randint(-30,30,1),np.random.randint(-30,30,1)]

            image=TF.affine(image, angle=0, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)     
            segmentation=TF.affine(segmentation,  angle=0, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fillcolor=None)

            aa=np.random.randint(-100,100,5)
            Threshold=0         
            if aa[0]<Threshold:
                gamma=np.random.randint(60,150,1)[0]*0.01
                image=TF.adjust_gamma(image, gamma, gain=1)

            if aa[2]<Threshold:
                saturation_factor=np.random.randint(60,150,1)*0.01
                image=TF.adjust_saturation(image, saturation_factor)
            if aa[3]<Threshold:
                contrast_factor=np.random.randint(80,120,1)*0.01
                image=TF.adjust_contrast(image, contrast_factor)              




        
     
    segmentation=np.array(segmentation)
    image=np.array(image)
        
    
    image,segmentation= CropCentre(image,segmentation,ImgSize)
   

    #print(image.shape)
    if image.ndim==2:
        image=image[:,:,np.newaxis]
        
    if segmentation.ndim==2:
        segmentation=segmentation[:,:,np.newaxis]  

    #print(image.shape)
    image= Normalise(image)
    #print(image.shape)
    segmentation= Normalise(segmentation)

    
    image=TF.to_tensor(image)    
    segmentation=TF.to_tensor(segmentation)
    #print(image.shape)

    return image, segmentation

def Normalise(Image):
    if isinstance(Image,(np.ndarray)): 
        return (Image-np.min(Image))/(np.max(Image)-np.min(Image)+0.0000001)
    elif isinstance(Image,(torch.Tensor)):
        return (Image-torch.min(Image))/(torch.max(Image)-torch.min(Image)+0.0000001)
    

    
    
    
class ReadTheDataset(data.Dataset):
    def __init__(self,opt, image_dir,Module):
        super(ReadTheDataset, self).__init__()
        
        ############ test set ######################

        self.Images_3d = [x for x in sorted(listdir(join(image_dir,"images") )) if is_np_file(x) ] 
        self.GTs_3d = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_np_file(x) ] 
        
        self.Images = [x for x in sorted(listdir(join(image_dir,"images") )) if is_png_image_file(x) ] 
        self.GTs = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_png_image_file(x) ] 

        
        self.Module=Module

        self.opt=opt


        self.image_dir = image_dir


    def __getitem__(self, index):
        

        
        if self.image_dir.endswith('test'):

            
            self.GTs_3d.sort()
            self.Images_3d.sort()  
            
            targets= np.load(join(self.image_dir,'1st_manual/',self.GTs_3d[index]))
            inputIms=np.load(join(self.image_dir,'images/',self.Images_3d[index])) 
                    
            
            #targets=Normalise(targets)
            #inputIms=Normalise(inputIms)
            assert(np.max(targets)<1.001, 'error 1')
            assert(np.max(inputIms)<1.001, 'error 2')
            
                
            inputIms,targets= CropCentre(inputIms,targets, self.opt.InputSize)
 

            if inputIms.ndim==2:
                inputIms=inputIms[:,:,None]
            if targets.ndim==2:
                targets=targets[:,:,None]


            inputIms=TF.to_tensor(inputIms)
            targets=TF.to_tensor(targets)

    
            
            view=[]
            ImNo=[]
            
        elif self.image_dir.endswith('training'):
            
            self.GTs.sort()
            self.Images.sort()
            
            
            
            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms)              

            #print(targets.shape,inputIms.shape)
            #targets=targets[0,:,:]
            #inputIms=inputIms[0,:,:]

            #print(self.Images[index],self.GTs[index])    
            inputIm2,targets2=ImageTransforms(inputIms, targets,self.opt.InputSize, training=True, Module=self.Module)

            #print(inputIm2.shape,targets2.shape)
            
            
            
            
            del inputIms
            del targets

            inputIms=inputIm2
            targets= targets2
            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=float(cc.split("_",3)[-1])
            
            
            ImNo=float(cc.split("_",3)[-2][2:])
            #print(Names,view,ImNo)
  
  
        elif self.image_dir.endswith('validation'):

            
            self.GTs.sort()
            self.Images.sort()      
            
            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms) 
            
           
            

            inputIm2,targets2=ImageTransforms(inputIms, targets,self.opt.InputSize, training=False)



            inputIms=inputIm2
            targets= targets2
            
            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=float(cc.split("_",3)[-1])
            
            
            ImNo=float(cc.split("_",3)[-2][2:])
           
 
        #pdb.set_trace()
    
        if torch.max(targets)>1:
            targets=targets/torch.max(targets)
            
        if torch.max(inputIms)>1:
            inputIms=inputIms/torch.max(inputIms)
            
            
        torch._assert( torch.max(targets)<1.01 , "target should be normalized")
        torch._assert( torch.max(inputIms)<1.01 , "input images should be normalized")
        

        return inputIms, targets,ImNo,view
    
    def __len__(self):
        if self.image_dir.endswith('test'):
            return len(self.GTs_3d)
            
        else:
            return len(self.GTs)
    
                
class Read_DRIVE_Dataset(data.Dataset):
    def __init__(self,opt, image_dir,Module):
        super(Read_DRIVE_Dataset, self).__init__()
        
        ############ test set ######################


        if opt.Dataset=='DRIVE':
            self.Images = [x for x in sorted(listdir(join(image_dir,"images") )) if is_tif_image_file(x) ] 
            self.GTs = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_gif_image_file(x) ] 
            self.Masks = [x for x in sorted(listdir(join(image_dir,"mask") )) if is_gif_image_file(x) ] 
        elif  opt.Dataset=='STARE':
            self.Images = [x for x in sorted(listdir(join(image_dir,"images") )) if is_ppm_image_file(x) ] 
            self.GTs = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_ppm_image_file(x) ] 
            self.Masks = self.GTs
        elif  opt.Dataset=='IOSTAR':
            self.Images = [x for x in sorted(listdir(join(image_dir,"images") )) if is_jpg_image_file(x) ] 
            self.GTs = [x for x in sorted(listdir(join(image_dir,"1st_manual") )) if is_tif_image_file(x) ] 
            self.Masks = self.GTs        
        self.Module=Module

        self.opt=opt


        self.image_dir = image_dir
        


    def __getitem__(self, index):
        

        
        if self.image_dir.endswith('test'):

            
            self.GTs.sort()
            self.Images.sort()  
            
            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            masks=load_img(join(self.image_dir,'mask/',self.Masks[index])) 
                    
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms)
            masks=Normalise(masks)
            assert(np.max(targets)<1.001, 'error 1')
            assert(np.max(inputIms)<1.001, 'error 2')
            
                
            inputIms,targets= CropCentre(inputIms,targets, self.opt.InputSize)
            masks,_= CropCentre(masks,masks, self.opt.InputSize)
 

            if inputIms.ndim==2:
                inputIms=inputIms[:,:,None]
            if targets.ndim==2:
                targets=targets[:,:,None]
            if masks.ndim==2:
                masks=masks[:,:,None]

            inputIms=TF.to_tensor(inputIms)
            targets=TF.to_tensor(targets)
            masks=TF.to_tensor(masks)

    
            view=[]    
        elif self.image_dir.endswith('training'):
            
            self.GTs.sort()
            self.Images.sort()
            

            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms)              

            #print(targets.shape,inputIms.shape)
            #targets=targets[0,:,:]
            #inputIms=inputIms[0,:,:]

            #print(self.Images[index],self.GTs[index])    
            inputIm2,targets2=ImageTransforms(inputIms, targets,self.opt.InputSize, training=True, Module=self.Module)

            #print(inputIm2.shape,targets2.shape)
            
            
            
            
            del inputIms
            del targets

            inputIms=inputIm2
            targets= targets2
            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=cc.split("_",3)[1][0]
            masks=[]
  
        elif self.image_dir.endswith('validation'):

            
            self.GTs.sort()
            self.Images.sort()      
            
            targets= load_img(join(self.image_dir,'1st_manual/',self.GTs[index]))
            inputIms=load_img(join(self.image_dir,'images/',self.Images[index])) 
            
            targets=Normalise(targets)
            inputIms=Normalise(inputIms) 
            
           
            

            inputIm2,targets2=ImageTransforms(inputIms, targets,self.opt.InputSize, training=False)



            inputIms=inputIm2
            targets= targets2
            
            
            Names=self.GTs[index]
            cc=Names.split(".",2)[0]
            view=cc.split("_",3)[1][0]
           
            masks=[]
        #pdb.set_trace()
    
        if torch.max(targets)>1:
            targets=targets/torch.max(targets)
            
        if torch.max(inputIms)>1:
            inputIms=inputIms/torch.max(inputIms)
            
            
        torch._assert( torch.max(targets)<1.01 , "target should be normalized")
        torch._assert( torch.max(inputIms)<1.01 , "input images should be normalized")
        

        return inputIms, targets,view,masks
    
    def __len__(self):

        return len(self.GTs)
  

def is_np_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_png_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])
def is_jpg_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_tif_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])
def is_gif_image_file(filename):
    return any(filename.endswith(extension) for extension in [".gif"])


def is_ppm_image_file(filename):
    return any(filename.endswith(extension) for extension in [".ppm"])
def load_img(filepath,colordim=1):
    if colordim==1:
        #img=PIL.ImageOps.grayscale(Image.open(filepath))
        #img = Image.open(filepath).convert('I')
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return np.array(img)
