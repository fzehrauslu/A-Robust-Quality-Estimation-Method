#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from math import sqrt
import pdb

from pytorch_wavelets import DWTForward, DWTInverse

import copy

class TMSNet_4Decoders(nn.Module):
    def __init__(self,W=4, WaveletDenoising=True, FeatureAdapt=True ):
        super(TMSNet_4Decoders, self).__init__()
        

       
        KernelSize=3
        Normalisation="Group" 
        Activation=F.relu
        NormalisationFirst=False
        WaveletDenoising=WaveletDenoising

        self.Decoder1=Decoder(FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self.Decoder2=Decoder(FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self.Decoder3=Decoder(FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self.Decoder4=Decoder(FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst)

        self.Encoder=Encoder(KernelSize, Normalisation,W, Activation,NormalisationFirst,WaveletDenoising)
        self._initialize_weights()
    def forward(self, x):

        x0, x1, x2,x3,x4, x1_H, x2_H,x3_H,x4_H= self.Encoder(x)

        xup1= self.Decoder1(x0.clone(), x1.clone(),  x2.clone(),x3.clone(),x4.clone(), copy.copy(x1_H), copy.copy(x2_H),copy.copy(x3_H),copy.copy(x4_H))
        xup2= self.Decoder2(x0.clone(), x1.clone(),  x2.clone(),x3.clone(),x4.clone(), copy.copy(x1_H), copy.copy(x2_H),copy.copy(x3_H),copy.copy(x4_H))
        
        xup3= self.Decoder3(x0.clone(), x1.clone(),  x2.clone(),x3.clone(),x4.clone(), copy.copy(x1_H), copy.copy(x2_H),copy.copy(x3_H),copy.copy(x4_H))
        xup4= self.Decoder4(x0.clone(), x1.clone(),  x2.clone(),x3.clone(),x4.clone(), copy.copy(x1_H), copy.copy(x2_H),copy.copy(x3_H),copy.copy(x4_H))
        
        
        
        return xup1,xup2,xup3,xup4
            

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 
                
class Encoder(nn.Module):
    def __init__(self, KernelSize, Normalisation,W, Activation,NormalisationFirst,WaveletDenoising=True):
        super(Encoder, self).__init__()

        
        self.denoising=WaveletDenoising
        
        if self.denoising:
            
            self.Unet_Layer0=ResidualLayer(1,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer1=ResidualLayer(W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer2=ResidualLayer(2*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer3=ResidualLayer(4*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer4=ResidualLayer(8*W,16*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)                 

            # denoising layers 
            self.Unet_Layer0_C1=ResidualLayer(W,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer0_C2=ResidualLayer(W,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer0_C3=ResidualLayer(W,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer0_C4=ResidualLayer(W,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)


            self.Unet_Layer1_C1=ResidualLayer(2*W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer1_C2=ResidualLayer(2*W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer1_C3=ResidualLayer(2*W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer1_C4=ResidualLayer(2*W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)

            self.Unet_Layer2_C1=ResidualLayer(4*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer2_C2=ResidualLayer(4*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer2_C3=ResidualLayer(4*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer2_C4=ResidualLayer(4*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)


            self.Unet_Layer3_C1=ResidualLayer(8*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer3_C2=ResidualLayer(8*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer3_C3=ResidualLayer(8*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
            self.Unet_Layer3_C4=ResidualLayer(8*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)


            self.WaveletPooling=DWTForward(J=1, mode='reflect', wave='haar') 
            self.WaveletUnPooling=DWTInverse( mode='reflect', wave='haar')
                          
        else:
            self.Unet_Layer0=ResidualLayer(1,W,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=1)
            self.Unet_Layer1=ResidualLayer(W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=2)
            self.Unet_Layer2=ResidualLayer(2*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=2)
            self.Unet_Layer3=ResidualLayer(4*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=2)
            self.Unet_Layer4=ResidualLayer(8*W,16*W,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=2)  

        self._initialize_weights()
        
    def waveletDenoising(self,x, Layer1,Layer2,Layer3,Layer4,onlyDenoising=False):
        
        
        x1,x1_H=self.WaveletPooling(x)
        
        x1=Layer1(x1)
        b,c,_,w,h=x1_H[0].shape
        
        
        x1_H[0][:,:,0,:,:]=Layer2(x1_H[0][:,:,0,:,:])
        x1_H[0][:,:,1,:,:]=Layer3(x1_H[0][:,:,1,:,:])
        x1_H[0][:,:,2,:,:]=Layer4(x1_H[0][:,:,2,:,:])
               

        x1D=self.WaveletUnPooling((x1,x1_H))+x
       
        if onlyDenoising:
            return x1D 
        else:
            x1,x1_H=self.WaveletPooling(x1D)
            return x1,x1_H,x1D   
        
        
    def encoderLayers(self,x): 
        
        if self.denoising:
            x0= self.Unet_Layer0(x) 

            if self.denoising:

                x0= self.waveletDenoising(x0, self.Unet_Layer0_C1,self.Unet_Layer0_C2,self.Unet_Layer0_C3,self.Unet_Layer0_C4, onlyDenoising=True)


            x1 = self.Unet_Layer1(x0)

            if self.denoising:  

                x1,x1_H,x1D= self.waveletDenoising(x1, self.Unet_Layer1_C1,self.Unet_Layer1_C2,self.Unet_Layer1_C3,self.Unet_Layer1_C4)

            else:
                x1,x1_H=self.WaveletPooling(x1)


            x2 = self.Unet_Layer2(x1)

            if self.denoising:  

                x2,x2_H,x2D= self.waveletDenoising(x2, self.Unet_Layer2_C1,self.Unet_Layer2_C2, self.Unet_Layer2_C3,self.Unet_Layer2_C4)   
            else:
                x2,x2_H=self.WaveletPooling(x2) 

            x3 = self.Unet_Layer3(x2)

            if self.denoising:  
                x3,x3_H,x3D= self.waveletDenoising(x3, self.Unet_Layer3_C1,self.Unet_Layer3_C2,self.Unet_Layer3_C3,self.Unet_Layer3_C4) 
            else:
                x3,x3_H=self.WaveletPooling(x3)  

            x4 = self.Unet_Layer4(x3)

            x4,x4_H=self.WaveletPooling(x4)


            return x0, x1, x2,x3,x4, x1_H, x2_H,x3_H,x4_H        
    
        else:
            
            x0= self.Unet_Layer0(x) 

            x1 = self.Unet_Layer1(x0)

            x2 = self.Unet_Layer2(x1)


            x3 = self.Unet_Layer3(x2)


            x4 = self.Unet_Layer4(x3)


            return x0, x1, x2,x3,x4 
        
        
    def forward(self,x):

        return self.encoderLayers(x) 
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_()          
                
                
class Decoder(nn.Module):
    def __init__(self,FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst):
        super(Decoder, self).__init__()

        
        self.decoderLayer=DecoderPerView(FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self.outputLayer=nn.Conv2d(W, 1,1)
        
        self._initialize_weights()
        
  
    def forward(self,x0, x1, x2,x3,x4, x1_H, x2_H,x3_H,x4_H ):

        xup= self.decoderLayer(x0.clone(), x1.clone(), x2.clone(),x3.clone(),x4.clone(), copy.copy(x1_H), copy.copy(x2_H),copy.copy(x3_H),copy.copy(x4_H)) 

        return self.outputLayer(xup)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 
                
                
class DecoderPerView(nn.Module):
    def __init__(self,FeatureAdapt,KernelSize, Normalisation,W, Activation,NormalisationFirst):
        super(DecoderPerView, self).__init__()


        self.WaveletUnPooling=DWTInverse( mode='reflect', wave='haar')

        
        #self.E1=ResidualLayer(8*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)        
        
        self.E2=ResidualLayer(4*W,2*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)

        self.E3=ResidualLayer(8*W,4*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)

        self.E4=ResidualLayer(16*W,8*W,KernelSize, Normalisation,W, Activation,NormalisationFirst)


        
        
        
        # for denoising original
        self.FeatureAdaptation0=FeatureAdaptation(W,W,W,True,Activation,KernelSize,Normalisation,NormalisationFirst)
        self.FeatureAdaptation1=FeatureAdaptation(2*W,2*W,W,False,Activation,KernelSize,Normalisation,NormalisationFirst)
        self.FeatureAdaptation2=FeatureAdaptation(4*W,4*W,W,False,Activation,KernelSize,Normalisation,NormalisationFirst)
        self.FeatureAdaptation3=FeatureAdaptation(8*W,8*W,W,False,Activation,KernelSize,Normalisation,NormalisationFirst)
        self.FeatureAdaptation4=FeatureAdaptation(16*W,16*W,W,True,Activation,KernelSize,Normalisation,NormalisationFirst)

             
        self.out= ResidualLayer(9*W,W,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self._initialize_weights()
        
  
    def forward(self,x0, x1, x2,x3,x4, x1_H, x2_H,x3_H,x4_H):

        x0= self.FeatureAdaptation0(x0,[])

        x1,x1_H= self.FeatureAdaptation1(x1,x1_H)
        x2,x2_H= self.FeatureAdaptation2(x2,x2_H)
        x3,x3_H= self.FeatureAdaptation3(x3,x3_H)
        x4= self.FeatureAdaptation4(x4,[])

        
        f1=self.WaveletUnPooling((x1,x1_H))
        f2=self.WaveletUnPooling((self.E2(self.WaveletUnPooling((x2,x2_H))),x1_H))
        f3=self.WaveletUnPooling((self.E2(self.WaveletUnPooling((self.E3(self.WaveletUnPooling((x3,x3_H))),x2_H))),x1_H))
        f4=self.WaveletUnPooling((self.E2(self.WaveletUnPooling((self.E3(self.WaveletUnPooling((self.E4(self.WaveletUnPooling((x4,x4_H))),x3_H))),x2_H))),x1_H))
    

        fs=torch.cat((x0,f1,f2,f3,f4), 1)

        return self.out(fs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_()                 
                
class FeatureAdaptation(nn.Module):
    def __init__(self,InputChannelNoEncoder,InputChannelNoDecoder,W,Without_HighFrequency=False,Activation= F.relu,KernelSize=3,Normalisation="Group",NormalisationFirst=False):
        super(FeatureAdaptation, self).__init__()
        
        self.Without_HighFrequency=Without_HighFrequency 
          
        
        self.OutChannel=InputChannelNoDecoder

        self.Unet_Layer8E= torch.nn.Sequential(ResidualLayer(InputChannelNoEncoder,InputChannelNoEncoder,KernelSize, Normalisation,W, Activation,NormalisationFirst),ResidualLayer(InputChannelNoEncoder,InputChannelNoDecoder,KernelSize, Normalisation,W, Activation,NormalisationFirst) )
        
        if  self.Without_HighFrequency!=True:
            Normalisation=False
            self.D1_1=ConvLayer(4*InputChannelNoEncoder,3*InputChannelNoDecoder,KernelSize, Normalisation,W, Activation,NormalisationFirst)
                 
        
        self._initialize_weights()
        
        
    def forward(self,x1,x1_H):
        x1=self.Unet_Layer8E(x1)
        ############################################
        
        if self.Without_HighFrequency !=True:
            if len(x1_H)>0:
                b,c,_,w,h=x1_H[0].shape

                x1_H[0]=self.D1_1(torch.cat((x1,x1_H[0].view(b,-1,w,h)),1)).view(b,self.OutChannel,3,w,h)#-x1[:,None,:,:,:]


                return x1, x1_H        
        
            
        else:
            return x1
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 
                
                


class ResidualLayer(nn.Module):
    def __init__(self,InChannelNo,OutChannelNo,KernelSize, Normalisation,W, Activation, NormalisationFirst=True,stride=1):
        super(ResidualLayer, self).__init__()
        self.Layer1= ConvLayer(InChannelNo,OutChannelNo,KernelSize, Normalisation,W, Activation,NormalisationFirst)
        self.Layer2= ConvLayer(OutChannelNo,OutChannelNo,KernelSize, Normalisation,W, Activation,NormalisationFirst,stride=stride)
        
        self.resLayer=ConvLayer(InChannelNo,OutChannelNo,1, Normalisation,W, Activation,NormalisationFirst,stride=stride)
        self._initialize_weights()
        
        
    def forward(self,x):
        
        x=self.Layer2(self.Layer1(x))+self.resLayer(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 


class ConvLayer(nn.Module):
    def __init__(self,InChannelNo,OutChannelNo,KernelSize, Normalisation,W, Activation,NormalisationFirst=True,stride=1):
        super(ConvLayer, self).__init__()
        
        padding=int((KernelSize-1)/2)
        
        self.Layer=nn.Conv2d(InChannelNo, OutChannelNo, KernelSize,padding=padding,stride=stride)         
        self.NormalisationFirst=NormalisationFirst
        self.Activation=Activation
        self.Normalisation=None
        
        
        if Normalisation=="Group":
            self.Normalisation=nn.GroupNorm(int(OutChannelNo/W), OutChannelNo)
        elif Normalisation=="Batch":
            self.Normalisation=nn.BatchNorm2d(OutChannelNo)            
        elif Normalisation=="Instance":
            self.Normalisation=nn.InstanceNorm2d(OutChannelNo)            
            

        self._initialize_weights()
        
        
    def forward(self,x):
        x=self.Layer(x)

        if self.Normalisation==None:
            x=self.Activation(x)
        else: 
            
            if self.NormalisationFirst:
                x=self.Activation(self.Normalisation(x))

            else:
                x=self.Normalisation(self.Activation(x))
        
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 


                
                