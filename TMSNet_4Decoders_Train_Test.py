#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#########################################################3
# thresh=threshold_otsu(vect)
#########################################################3



import torch
import torch.nn as nn 
import numpy as np
import torchvision
import os
import pdb
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL
from skimage.filters import threshold_otsu
from medpy.metric.binary import hd, assd, dc,jc
from skimage.metrics import structural_similarity as ssim
from medpy.metric.image import mutual_information
from sklearn.metrics.pairwise import cosine_similarity as cos
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score as kappa
import torchmetrics
import json

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def SaveImagesForExperiment(vect,name='_n_.png'):
    for num in range(4):
        torchvision.utils.save_image(torch.from_numpy(1-1.*vect[:,num,None,:,:]),str(num)+name)



def randomNoise(size):
    a= np.random.uniform(low=-0.1, high=0.1, size=size)
    b=np.random.uniform(low=-0.1, high=0.1, size=size)
    
    while(cos(a,b)>0.01):
        a= np.random.uniform(low=-0.1, high=0.1, size=size)
        b=np.random.uniform(low=-0.1, high=0.1, size=size)
    return a,b
    
    
    
crossEntropy=nn.BCEWithLogitsLoss() 
cosSimilarity=nn.CosineSimilarity(dim=1)
crossEntropy2=nn.BCEWithLogitsLoss(reduce=False)


    
    
def CalculateAggregatedOutput(vect,method='mean',thresh=2):
    
    
    if method=='mean':
        
        thresh=threshold_otsu(vect)
        vect=np.float32(vect>thresh)
        Prob=np.mean(vect,axis=1)[:,None,:,:]
        ###########################################
        thresh=threshold_otsu(Prob)
        
        ###########################################
    
    
        vectBinary=np.int8(Prob>thresh)
        
        
        
    
    elif method=='majorityVoting':   
        thresh=threshold_otsu(vect)
        vect=np.int8(vect>thresh)
        vectBinary=np.int8(np.sum(vect,axis=1)>thresh)[:,None,:,:] #1

    return vectBinary





def CalculateSimilarity(vect,input,similarity='Dice',method='mean',batch=[],noise=False):
    b,c,w,h=vect.shape
    vect=vect.cpu().numpy()
    input=input.cpu().numpy()
    
    #################
    
    vectOriginal=np.float32(vect).copy()
    if noise:
        from scipy.stats import ortho_group
        aa=ortho_group.rvs(dim= 4)
        OrtMatrix=np.repeat(aa[None,:,None,:],b,0)
        
        OrtMatrix=np.repeat(OrtMatrix,w,2)
        OrtMatrix=np.repeat(OrtMatrix,int(h/4),-1)

        
        vectOriginal=vectOriginal+0.05*OrtMatrix
    ##################3
    
    #### binary vect and vectAgg
    vectAgg=CalculateAggregatedOutput(vect,method=method)
    #pdb.set_trace()
    results=PerDecoderSimilarities(vectOriginal,similarity,vectAgg,input)
    return  results



def PerDecoderSimilarities(vect,similarity,meanVect,input):
    
    vect=(vect-np.min(vect))/(np.max(vect)-np.min(vect)+0.00001)
    
    if similarity=='Dice':
        thresh=threshold_otsu(vect)
        vect=np.int8(vect>thresh)      
        
        results=[dc(meanVect.ravel()[None,:],vect[:,0,:,:].ravel()[None,:]),dc(meanVect.ravel()[None,:],vect[:,1,:,:].ravel()[None,:]),dc(meanVect.ravel()[None,:],vect[:,2,:,:].ravel()[None,:]),dc(meanVect.ravel()[None,:],vect[:,3,:,:].ravel()[None,:])]
        

    elif similarity=='cosine':  

        results=[cos(meanVect.ravel()[None,:],vect[:,0,:,:].ravel()[None,:]).item(),cos(meanVect.ravel()[None,:],vect[:,1,:,:].ravel()[None,:]).item(),cos(meanVect.ravel()[None,:],vect[:,2,:,:].ravel()[None,:]).item(),cos(meanVect.ravel()[None,:],vect[:,3,:,:].ravel()[None,:]).item() ]
        
         
    return  results   
    



    

    





def train(TranPara,opt,model,epoch,optimizer):
    

    model.train()

    trainLoss=0

    valLoss=0
        
     
    
    DataloadersCell=[ TranPara.training_data_loader_All,TranPara.training_data_loader_1, TranPara.training_data_loader_2,TranPara.training_data_loader_3,TranPara.training_data_loader_4] # 
    TrainedParts=["encoder ",'decoder 1','decoder 2','decoder 3' ,'decoder 4']
    
    print("training starts")
    it=0
    for training_data_loader in DataloadersCell:
        
        for itera, batch in enumerate(training_data_loader):


            input=batch[0].type(torch.FloatTensor).to(TranPara.device)
            
            target=batch[1].type(torch.FloatTensor).to(TranPara.device)

            ########################## 
            if training_data_loader==TranPara.training_data_loader_All:
                if itera ==0:
                    model.Encoder.requires_grad = True
                    model.Decoder1.requires_grad = True
                    model.Decoder2.requires_grad = True
                    model.Decoder3.requires_grad = True
                    model.Decoder4.requires_grad = True
                    


                else:
                    model.Encoder.requires_grad = True
                    model.Decoder1.requires_grad = False
                    model.Decoder2.requires_grad = False
                    model.Decoder3.requires_grad = False
                    model.Decoder4.requires_grad = False

                    
                    
            elif training_data_loader==TranPara.training_data_loader_1:
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = True
                model.Decoder2.requires_grad = False
                model.Decoder3.requires_grad = False
                model.Decoder4.requires_grad = False
                
                
                
                
            elif training_data_loader==TranPara.training_data_loader_2:
                
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = False
                model.Decoder2.requires_grad = True
                model.Decoder3.requires_grad = False                
                model.Decoder4.requires_grad = False
                
                
            elif training_data_loader==TranPara.training_data_loader_3:
                
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = False
                model.Decoder2.requires_grad = False
                model.Decoder3.requires_grad = True                
                model.Decoder4.requires_grad = False
                
                
            elif training_data_loader==TranPara.training_data_loader_4:
                
                model.Encoder.requires_grad = False
                model.Decoder1.requires_grad = False
                model.Decoder2.requires_grad = False
                model.Decoder3.requires_grad = False                
                model.Decoder4.requires_grad = True
                
                
            
            
            if training_data_loader==TranPara.training_data_loader_All:


                
                loss=(crossEntropy(model(input)[0],target)+crossEntropy(TF.rotate(model(TF.rotate(input, angle=-90))[1],angle=90),target)+crossEntropy(TF.rotate(model(TF.rotate(input, angle=90))[-1],angle=-90),target)+ crossEntropy(TF.rotate(model(TF.rotate(input, angle=180))[2],angle=-180),target))/4.
            
                
            
            elif training_data_loader==TranPara.training_data_loader_1:
                prediction1=model(input)[0]
                loss=crossEntropy(prediction1,target)
                
                
            elif training_data_loader==TranPara.training_data_loader_2:
                
                input=TF.rotate(input, angle=-90)
                target=TF.rotate(target, angle=-90)

                prediction1=model(input)[1]                
                loss=crossEntropy(prediction1,target)
                
            elif training_data_loader==TranPara.training_data_loader_3:

                input=TF.rotate(input, angle=180)
                target=TF.rotate(target, angle=180)
                
                prediction1=model(input)[2]
                loss= crossEntropy(prediction1,target)
                
            elif training_data_loader==TranPara.training_data_loader_4:
                
                input=TF.rotate(input, angle=90)
                target=TF.rotate(target, angle=90)
                
                prediction1=model(input)[-1]                
                loss=crossEntropy(prediction1,target)
                
         
            del target       
            del input             
            torch.cuda.empty_cache()


            trainLoss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            np.random.seed(TranPara.initial_seed + epoch)

        print("                                ")        
        print(TrainedParts[it]+' was trained')        
        print("                                ") 
        it+=1
        
        
        
        
    print("validation starts")
    
    DataloadersCell=[TranPara.val_data_loader_All,TranPara.val_data_loader_1,TranPara.val_data_loader_2, TranPara.val_data_loader_3,TranPara.val_data_loader_4]
    TrainedParts=["encoder ",'decoder 1','decoder 2','decoder 3','decoder 4' ]

    with torch.no_grad():
        model.eval()
        
        for val_data_loader in DataloadersCell:

            for _, batch in enumerate(val_data_loader):

                

                input=batch[0].type(torch.FloatTensor).to(TranPara.device)
                target=batch[1].type(torch.FloatTensor).to(TranPara.device)
                

            
                if val_data_loader==TranPara.val_data_loader_All:
                    loss=(crossEntropy(model(input)[0],target)+crossEntropy(TF.rotate(model(TF.rotate(input, angle=-90))[1],angle=90),target)+crossEntropy(TF.rotate(model(TF.rotate(input, angle=90))[-1],angle=-90),target)+ crossEntropy(TF.rotate(model(TF.rotate(input, angle=180))[2],angle=-180),target))/4.

                

                elif val_data_loader==TranPara.val_data_loader_A:

                    prediction1=model(input)[0]
                    loss= crossEntropy(prediction1,target)
                elif val_data_loader==TranPara.val_data_loader_C:
                    input=TF.rotate(input, angle=-90)
                    target=TF.rotate(target, angle=-90)                    
                    prediction1=model(input)[1]                
                    loss= crossEntropy(prediction1,target)
                elif val_data_loader==TranPara.val_data_loader_S:

                    input=TF.rotate(input, angle=180)
                    target=TF.rotate(target, angle=180)               
                    prediction1=model(input)[2]
                    loss= crossEntropy(prediction1,target)
                    
                elif val_data_loader==TranPara.val_data_loader_Z:

                    input=TF.rotate(input, angle=90)
                    target=TF.rotate(target, angle=90)

                    prediction1=model(input)[-1]                
                    loss= crossEntropy(prediction1,target)

               
            
                valLoss +=loss.item()
            
            ###########################################################




    trainLoss=trainLoss/(len(TranPara.training_data_loader_All)+ len(TranPara.training_data_loader_1)+len(TranPara.training_data_loader_2)+len(TranPara.training_data_loader_3)+len(TranPara.training_data_loader_4))
    valLoss=valLoss/(len(TranPara.val_data_loader_All)+len(TranPara.val_data_loader_1)+ len(TranPara.val_data_loader_2)+len(TranPara.val_data_loader_3)+len(TranPara.val_data_loader_4))



    print("===> Epoch {} Complete: Avg. Unet Loss: {:.4f}".format(epoch, trainLoss ))
    print("===> Epoch {} Complete: Avg. Unet Validation Loss: {:.4f}".format(epoch, valLoss ))

    return model, optimizer,trainLoss, valLoss





        



    
    
def measure_latency_cpu_usage(model, test_inputs):
    
    import time
    import psutil
   
    process = psutil.Process()
    cpu_start = process.cpu_percent()
    start = time.time()
    predictions = model.predict(test_inputs)
    end = time.time()
    cpu_end = process.cpu_percent()
    latency = end - start
    cpu_usage = cpu_end - cpu_start
    return latency, cpu_usage    
    
def test(TranPara,opt,model, device,epsilon=0,method='mean',noise=False,TestDataset=''): 

       
    CosineSimilarities=[]
    DiceSimilarities=[]
    dice=[]
    Jaccard=[]
    hfDistance=[]
    ASSD=[]

        
    
    model.eval()
    totalloss = 0
    ImgNo=0

    for batch in TranPara.testing_data_loader:
        
        batch[0]=batch[0].permute(1,0,2,3)
        batch[1]=batch[1].permute(1,0,2,3)
       
        ImgNo=ImgNo+1
       
        


        batch[1]=batch[1].to(device)
        batch[0]=batch[0].type(torch.FloatTensor).to(device)

        b,c,w,h=batch[0].shape
        if w%32 !=0 or h%32 !=0:
            batch0_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch0_AA[:,:,:w,:h]=batch[0].clone()
            batch1_AA=torch.zeros(b,c,opt.InputSize,opt.InputSize).to(device)
            batch1_AA[:,:,:w,:h]=batch[1].clone()                

        else:
            batch0_AA=batch[0].clone()
            batch1_AA=batch[1].clone()

            
        ########################################

        
        if noise : 
            if opt.Dataset=='Atrium2018' or opt.Dataset=='Atrium2013':
                batch0_AA= AddRicianNoise(batch0_AA,TranPara.device,std=epsilon,mean=0)
            
            elif opt.Dataset=='DRIVE':
                batch0_AA=AddGaussianNoise(batch0_AA,TranPara.device,std=epsilon,mean=0)
            
   
        with torch.no_grad(): 
            vect_1= PredictVolume(opt,batch0_AA,model,ImgNo,TranPara.device,DecoderNo=1)
            
            vect_2= PredictVolume(opt,TF.rotate(batch0_AA, angle=-90),model,ImgNo,TranPara.device,DecoderNo=2)
            vect_2=TF.rotate(vect_2, angle=90)
            
            
            vect_3= PredictVolume(opt,TF.rotate(batch0_AA, angle=180),model,ImgNo,TranPara.device,DecoderNo=3)
            vect_3=TF.rotate(vect_3, angle=-180)

            
            vect_4= PredictVolume(opt,TF.rotate(batch0_AA, angle=90),model,ImgNo,TranPara.device,DecoderNo=4)
            vect_4=TF.rotate(vect_4, angle=-90)
                
        vect=torch.cat((vect_1, vect_2,vect_3,vect_4),1)
        
        
        
        #######################  calculate cosine similarities between decoder outputs ################       
        batch[1]=batch[1].cpu()
        batch[0]=batch[0].cpu()


        
        CosineSimilarities.append([CalculateSimilarity(vect.clone(), batch0_AA,similarity='cosine',method=method,batch=batch,noise=noise)])
        DiceSimilarities.append([CalculateSimilarity(vect.clone(), batch0_AA,similarity='Dice',method=method,batch=batch,noise=noise)])       

      
        


        vect=torch.from_numpy( CalculateAggregatedOutput(vect.cpu().numpy().copy(),method=method))

        predictions,dice,Jaccard,hfDistance,ASSD =CalculatePerformanceMetrics(opt,vect.clone(),batch,dice,Jaccard,hfDistance,ASSD,ImgNo)
        

    data={"dice":dice, "hfDistance":hfDistance,"ASSD":ASSD,"Jaccard":Jaccard,'CosineSimilarities':CosineSimilarities,'DiceSimilarities':DiceSimilarities}
    #testWithAdversarialNoise
    print ("        ")
    print (" epsilon  ",epsilon)

    print ("        ")
    print(np.mean(ASSD),np.mean(hfDistance),np.mean(dice),np.mean(Jaccard))
    print(np.std(ASSD),np.std(hfDistance),np.std(dice),np.std(Jaccard))
    with open(os.path.join(opt.PathToSaveTrainedModels,'PerformanceMetrics_tested_with_'+TestDataset+'.json'), 'w') as fp:
        json.dump(data, fp)   
            



    
    
     

    
def PredictVolume(opt,inputN,model,ImgNo,device,DecoderNo=1,NoSigmoid=False):
    #pdb.set_trace()
    b,c,w,h=inputN.shape
    
        
    
    if w%32 !=0 or h%32 !=0:
        input=torch.zeros((b,c,opt.InputSize,opt.InputSize)).to(device)
        input[:,:,:w,:h]=inputN
    else:
        input=inputN
     
    del inputN
    torch.cuda.empty_cache()
     
    ChunkSize=20
    for ChunckNo in range(ChunkSize,input.shape[0]+ChunkSize,ChunkSize):

        if  ChunckNo==ChunkSize:  

            if DecoderNo==1:
                vect=model(input[:ChunckNo,:,:,:])[0] 
            elif DecoderNo==2:
                vect=model(input[:ChunckNo,:,:,:])[1] 
            elif DecoderNo==3:
                vect=model(input[:ChunckNo,:,:,:])[2] 
            elif DecoderNo==4:
                vect=model(input[:ChunckNo,:,:,:])[3]                 
            if NoSigmoid==False:
                vect=torch.sigmoid(vect)
        else:


            if NoSigmoid==False:
                if DecoderNo==1:          
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[0])),0)
                elif DecoderNo==2:
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[1])),0)
                elif DecoderNo==3:
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[2])),0)  
                    
                elif DecoderNo==4:
                    vect=torch.cat((vect,torch.sigmoid(model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[3])),0)                      
                    
            else:
                if DecoderNo==1:          
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[0]),0)
                elif DecoderNo==2:
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[1]),0)
                elif DecoderNo==3:
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[2]),0)  

                elif DecoderNo==4:
                    vect=torch.cat((vect,model(input[ChunckNo-ChunkSize:ChunckNo,:,:,:])[3]),0)                
    if w%32 !=0 or h%32 !=0:
        vectR = vect[:,:,:w,:h]
    else:
        vectR = vect
    del vect
    torch.cuda.empty_cache()        
      
    return vectR
    
def CalculatePerformanceMetrics(opt,vect,batch,dice,Jaccard,hfDistance,ASSD,ImgNo):
    NuOfSlices=vect.shape[0]
    w=vect.shape[2]
    
    
    if opt.Dataset=='DRIVE':
        predictions=vect[0,:,:,:]
        indices=np.where(batch[-1].cpu().numpy().flatten()>0)[0]
        vect=vect.flatten()[indices]
        

        vectBinarised=np.reshape(np.array(vect),(-1)) #.cpu().numpy()
        GT=batch[1].flatten()[indices]
    
        thresh=threshold_otsu(vectBinarised)
        vectBinarised=np.int8(vectBinarised>thresh)
        
        GTvect=np.reshape(GT.numpy(),(-1))


        dice.append(dc(GTvect,vectBinarised))
        Jaccard.append(jc(GTvect,vectBinarised))

        hfDistance.append(1000)
        ASSD.append(1000)

            
            
    else:
        
         
        vectBinarised=np.reshape(np.array(vect),(-1)) 
        GT=batch[1]
        
        
        
        thresh=threshold_otsu(vectBinarised)
        vectBinarised=np.int8(vectBinarised>thresh)
        predictions=np.reshape(vectBinarised,(NuOfSlices,w,w))


        vectBinarised=np.reshape(predictions,(-1))
        GTvect=np.reshape(GT.numpy(),(-1))

        #pdb.set_trace()
        dice.append(dc(GTvect,vectBinarised))
        Jaccard.append(jc(GTvect,vectBinarised))


        if opt.Dataset=='Atrium2013' or opt.Dataset=='Atrium2018' or opt.Dataset=='Atrium':
            if opt.Dataset=='Atrium2013':

                voxelspacing=(2.7,1.25,1.25)             

            elif opt.Dataset=='Atrium2018':
                voxelspacing=(0.625,0.625*2,0.625*2)

            if np.sum(predictions) > 0:  
                hfDistance.append(hd(predictions,np.reshape(GTvect,(NuOfSlices,w,w)),voxelspacing=voxelspacing))
                ASSD.append(assd(predictions,np.reshape(GTvect,(NuOfSlices,w,w)),voxelspacing=voxelspacing))
            else:
                hfDistance.append(1000)
                ASSD.append(1000)
                print('No masks found')
        else:

            hfDistance.append(1000)
            ASSD.append(1000)
            print('No masks found')

    print("dice",dice[-1])
    print("Jaccard",Jaccard[-1])
    print("assd",ASSD[-1])
    print("hd",hfDistance[-1])

    return predictions,dice,Jaccard,hfDistance,ASSD

    


    
def AddRicianNoise(tensor,device,std=10,mean=0):

    
    #def AddRicianNoise(tensor,device,std,mean):
    b,c,w,h=tensor.shape
    noise1=(torch.randn(tensor.size())* std/100. + mean).to(device)
    noise2=(torch.randn(tensor.size())* std/100. + mean).to(device)
    #pdb.set_trace()
    tensor= torch.sqrt((tensor + noise1) ** 2 + noise2 ** 2)
    
    
    
    reshapedTensor=torch.reshape(tensor,(b,c,w*h))
    
    return  torch.clamp(tensor, 0, 1)

def AddGaussianNoise(tensor,device,std=10,mean=0):

    
    b,c,w,h=tensor.shape
    noise1=(torch.randn(tensor.size())* std/100. + mean).to(device)
    #pdb.set_trace()
    tensor= tensor + noise1
    
    reshapedTensor=torch.reshape(tensor,(b,c,w*h))
    
    return  torch.clamp(tensor, 0, 1)    