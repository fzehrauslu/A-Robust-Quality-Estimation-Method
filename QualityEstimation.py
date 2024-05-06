
import numpy as np
import random
import os
from os.path import  join
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import ExponentialLR as ExponentialLR

from sklearn.metrics import accuracy_score
import torchvision
import matplotlib.pyplot as plt

import medpy
import PIL
import json

from TMSNet_4Decoders_Train_Test import train, test
from TMSNet_4Decoders import TMSNet_4Decoders
import pdb

#import seaborn as snb


import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy

def DoAll(Dataset,mainPath,mainPathTraining,mainPathTest,Attacks,ignoreZero):
    QualityEstimation(Dataset,mainPathTraining,Attacks=Attacks,ignoreZero=ignoreZero)
    QualityEstimation(Dataset,mainPathTest,Attacks=Attacks,ignoreZero=ignoreZero)
    QualityEstimationCorrected(Dataset,mainPath,mainPathTraining,mainPathTest,Attacks=Attacks,ignoreZero=ignoreZero)

    
def EstimateNewValues(x,a,b,squeeze=False,KeepNegatives=True):
    if len(x)>0:
        if squeeze:
            MeanCosineSimilarities=np.squeeze(x*a+b)
        else:
            MeanCosineSimilarities=x*a+b

        if not KeepNegatives:
            inds=np.where(MeanCosineSimilarities<0)[0]   
            MeanCosineSimilarities[inds]=0.
    else:
        MeanCosineSimilarities=x
    return MeanCosineSimilarities
        
    
def RegressionModel(x,y):
    ## regression Model
    #pdb.set_trace()
    x=np.array(np.mean(x,axis=(1,2))).ravel()[:,None]
    y=np.array(y).ravel()[:,None]

    from sklearn.linear_model import LinearRegression
    #pdb.set_trace()
    model = LinearRegression().fit(x, y)
    b,a=model.intercept_,model.coef_
    return a[0],b

def ReadJsonFiles(mainPath,Dataset,Attacks,returnDetails=False):
    diceS1=[]
    jaccardS1=[]
    CosineSimilarities1=[]
    DiceSimilarities1=[]

    
    diceS2=[]
    jaccardS2=[]
    CosineSimilarities2=[]
    DiceSimilarities2=[]

    diceS3=[]
    jaccardS3=[]
    CosineSimilarities3=[]
    DiceSimilarities3=[]

   
    
    diceS4=[]
    jaccardS4=[]
    CosineSimilarities4=[]
    DiceSimilarities4=[]



    diceS5=[]
    jaccardS5=[]
    CosineSimilarities5=[]
    DiceSimilarities5=[]



    for AttackType in Attacks:#['Rician']: 
        
        if AttackType=='Rician':
            noiseRange=[5,10,15,20,25] #10,15,20,25

        elif AttackType=='Gaussian':
            noiseRange=[1,5,10,15,20,25] #10,15,20,25            
        else:
            noiseRange=[0.01,0.02,0.03,0.04]

        for nR in noiseRange:
            fileName=os.path.join(mainPath,'PerformanceMetrics_'+AttackType+'_'+str(nR)+'.json')

            if os.path.exists(fileName):
                print(fileName, 'found')
                f = open(fileName)
                data = json.load(f)
                diceS1+=data["dice"]
                CosineSimilarities1+=data["CosineSimilarities"]

                jaccardS1+= data["Jaccard"]
                DiceSimilarities1+=data["DiceSimilarities"]



                Briers1+= data["Biers"]
                ECEs1+= data["ECEs"]
                MCEs1+= data["MCEs"]
                LLs1+= data["LLs"]

                Entropies1+= data["MeanEntropys"]


    if Dataset=='Atrium2013':
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities2=data2["CosineSimilarities"]
            DiceSimilarities2=data2["DiceSimilarities"]            
            diceS2=data2["dice"]
            jaccardS2=data2["Jaccard"]

            Briers2= data2["Biers"]
            ECEs2= data2["ECEs"]
            MCEs2= data2["MCEs"]
            LLs2= data2["LLs"]
            Entropies2= data2["MeanEntropys"]



        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2013.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2013.json')) #
            data2 = json.load(f2)
            CosineSimilarities3=data2["CosineSimilarities"]
            DiceSimilarities3=data2["DiceSimilarities"]  


            diceS3=data2["dice"]
            jaccardS3=data2["Jaccard"]

            Briers3= data2["Biers"]
            ECEs3= data2["ECEs"]
            MCEs3= data2["MCEs"]
            LLs3= data2["LLs"]
            Entropies3= data2["MeanEntropys"]
        #pdb.set_trace()    
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2018.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2018.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities4=data2["CosineSimilarities"]
            DiceSimilarities4=data2["DiceSimilarities"]  


            diceS4=data2["dice"]
            jaccardS4=data2["Jaccard"]

            Briers4= data2["Biers"]
            ECEs4= data2["ECEs"]
            MCEs4= data2["MCEs"]
            LLs4= data2["LLs"]
            Entropies4= data2["MeanEntropys"]
            
    elif Dataset=='PrivateDataset':
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities2=data2["CosineSimilarities"]
            DiceSimilarities2=data2["DiceSimilarities"]            
            diceS2=data2["dice"]
            jaccardS2=data2["Jaccard"]

            Briers2= data2["Biers"]
            ECEs2= data2["ECEs"]
            MCEs2= data2["MCEs"]
            LLs2= data2["LLs"]
            Entropies2= data2["MeanEntropys"]



        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2013.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2013.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities3=data2["CosineSimilarities"]
            DiceSimilarities3=data2["DiceSimilarities"]  


            diceS3=data2["dice"]
            jaccardS3=data2["Jaccard"]

            Briers3= data2["Biers"]
            ECEs3= data2["ECEs"]
            MCEs3= data2["MCEs"]
            LLs3= data2["LLs"]
            Entropies3= data2["MeanEntropys"]
            
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2018.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_Atrium2018.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities4=data2["CosineSimilarities"]
            DiceSimilarities4=data2["DiceSimilarities"]  


            diceS4=data2["dice"]
            jaccardS4=data2["Jaccard"]

            Briers4= data2["Biers"]
            ECEs4= data2["ECEs"]
            MCEs4= data2["MCEs"]
            LLs4= data2["LLs"]
            Entropies4= data2["MeanEntropys"]  
            
            
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_Drive.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_Drive.json')) 
            data2 = json.load(f2)
            CosineSimilarities5=data2["CosineSimilarities"]
            DiceSimilarities5=data2["DiceSimilarities"]  


            diceS5=data2["dice"]
            jaccardS5=data2["Jaccard"]

            Briers5= data2["Biers"]
            ECEs5= data2["ECEs"]
            MCEs5= data2["MCEs"]
            LLs5= data2["LLs"]
            Entropies5= data2["MeanEntropys"]              

    elif Dataset=='DRIVE':
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_DRIVE.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_DRIVE.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities2=data2["CosineSimilarities"]
            DiceSimilarities2=data2["DiceSimilarities"]            
            diceS2=data2["dice"]
            jaccardS2=data2["Jaccard"]

            Briers2= data2["Biers"]
            ECEs2= data2["ECEs"]
            MCEs2= data2["MCEs"]
            LLs2= data2["LLs"]
            Entropies2= data2["MeanEntropys"]



        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_STARE.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_STARE.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities3=data2["CosineSimilarities"]
            DiceSimilarities3=data2["DiceSimilarities"]  


            diceS3=data2["dice"]
            jaccardS3=data2["Jaccard"]

            Briers3= data2["Biers"]
            ECEs3= data2["ECEs"]
            MCEs3= data2["MCEs"]
            LLs3= data2["LLs"]
            Entropies3= data2["MeanEntropys"]   
            
            
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_IOSTAR.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_IOSTAR.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities4=data2["CosineSimilarities"]
            DiceSimilarities4=data2["DiceSimilarities"]  


            diceS4=data2["dice"]
            jaccardS4=data2["Jaccard"]

            Briers4= data2["Biers"]
            ECEs4= data2["ECEs"]
            MCEs4= data2["MCEs"]
            LLs4= data2["LLs"]
            Entropies4= data2["MeanEntropys"]               
        if os.path.exists(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')):

            f2 = open(os.path.join(mainPath,'PerformanceMetrics_tested_with_PrivateDataset.json')) #PerformanceMetrics_testedWith2018_2
            data2 = json.load(f2)
            CosineSimilarities5=data2["CosineSimilarities"]
            DiceSimilarities5=data2["DiceSimilarities"]  


            diceS5=data2["dice"]
            jaccardS5=data2["Jaccard"]

            Briers5= data2["Biers"]
            ECEs5= data2["ECEs"]
            MCEs5= data2["MCEs"]
            LLs5= data2["LLs"]
            Entropies5= data2["MeanEntropys"]     
            
            
       
    diceS=np.array(diceS1+diceS2+diceS3+diceS4+diceS5)
    jaccardS=np.array(jaccardS1+jaccardS2+jaccardS3+jaccardS4+jaccardS5)

    CosineSimilarities=np.array(CosineSimilarities1+CosineSimilarities2+CosineSimilarities3+CosineSimilarities4+CosineSimilarities5)
    DiceSimilarities=np.array(DiceSimilarities1+DiceSimilarities2+DiceSimilarities3+DiceSimilarities4+DiceSimilarities5)




    Entropies= np.array(Entropies1+Entropies2+Entropies3+Entropies4+Entropies5)
    if not returnDetails:
        return diceS,jaccardS,CosineSimilarities,DiceSimilarities

    else:
        return DiceSimilarities1,DiceSimilarities2,DiceSimilarities3,DiceSimilarities4,DiceSimilarities5,CosineSimilarities1,CosineSimilarities2,CosineSimilarities3,CosineSimilarities4,CosineSimilarities5,diceS1,diceS2,diceS3,diceS4,diceS5

    

        

def QualityEstimationCorrected(Dataset,mainPath,mainPathTraining,mainPathTest,Attacks=['Rician'],ignoreZero=False): 

    
    
    
    
    #Dataset='2013'

    #mainPath='/home/oem/Desktop/TMS-Net/ekler/TrainedModels/2013_multipleDecoders'

    ## trainingSet
    diceS,jaccardS,CosineSimilarities,DiceSimilarities=ReadJsonFiles(mainPathTraining,Dataset, Attacks)
    if ignoreZero:
        indCos=np.where(np.mean(CosineSimilarities,axis=(1,2))>0.05)[0]
        indDice=np.where(np.mean(DiceSimilarities,axis=(1,2))>0.05)[0]

        a_cos,b_cos=RegressionModel(CosineSimilarities[indCos,:,:],diceS[indCos])
        a_dice,b_dice=RegressionModel(DiceSimilarities[indDice,:,:],diceS[indDice])
    else:
        a_cos,b_cos=RegressionModel(CosineSimilarities,diceS)
        a_dice,b_dice=RegressionModel(DiceSimilarities,diceS)        

    diceS,jaccardS,CosineSimilarities,DiceSimilarities=ReadJsonFiles(mainPathTest,Dataset,Attacks)
    

    
    np.savez(os.path.join(mainPathTest,'RegressionParameters.npz'),a_cos,b_cos,a_dice,b_dice)
    x=np.array(np.mean(CosineSimilarities,axis=(1,2))).ravel()#[:,None]
    
    MeanCosineSimilarities=EstimateNewValues(x,a_cos,b_cos,squeeze=True)
    
    
    #MeanCosineSimilarities=np.squeeze(x*a_cos+b_cos)
    
    x=np.array(np.mean(DiceSimilarities,axis=(1,2))).ravel()#[:,None]
    MeanDiceSimilarities=EstimateNewValues(x,a_dice,b_dice,squeeze=True)
    
    #MeanDiceSimilarities=np.squeeze(x*a_dice+b_dice)


    import scipy
    from scipy import stats    
    
    PerformMetric=np.array(diceS) # diceS
    
    t=0.7
    ## calculate accuracy, AUC, MAE, r

    print('linear regressionParameters for cosine - dice', a_cos,b_cos)
    print('linear regressionParameters for dice - dice', a_dice,b_dice)

    print('accuracy cosine - dice', accuracy_score(PerformMetric>t, MeanCosineSimilarities>t))
    print('accuracy dice - dice', accuracy_score(PerformMetric>t, MeanDiceSimilarities>t))

    if ignoreZero:
    
        indCos=np.where(MeanCosineSimilarities>0.05)[0]
        indDice=np.where(MeanDiceSimilarities>0.05)[0]
        
        
    if ignoreZero:
        MeanCosineCorr=stats.pearsonr(MeanCosineSimilarities[indCos],PerformMetric[indCos])
        print('pearson correlation_mean',MeanCosineCorr)
    else:
        MeanCosineCorr=stats.pearsonr(MeanCosineSimilarities,PerformMetric)
        print('pearson correlation_mean',MeanCosineCorr)


    if ignoreZero:
        MeanDiceCorr=stats.pearsonr(MeanDiceSimilarities[indDice],PerformMetric[indDice])

        print('pearson correlation_Dicemean',MeanDiceCorr)
        
    else:
        MeanDiceCorr=stats.pearsonr(MeanDiceSimilarities,PerformMetric)
        print('pearson correlation_Dicemean',MeanDiceCorr)


        
    y={'classificationAccuracy_cosine_dice_t_0_7':accuracy_score(PerformMetric>t, MeanCosineSimilarities>t),'classificationAccuracy_dice_dice_t_0_7':accuracy_score(PerformMetric>t, MeanDiceSimilarities>t),"cosineMeanCorrelation":MeanCosineCorr, "DiceMeanCorrelation":MeanDiceCorr,'meanCosSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,MeanCosineSimilarities), 'meanDiceSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,MeanDiceSimilarities), }
    
    
    with open(os.path.join(mainPathTest, 'QualityEstimation.json'), 'r') as fp:
        file_data = json.load(fp)
        
        
    file_data.update(y)
    with open(os.path.join(mainPathTest, 'QualityEstimation.json'), 'w+') as fp:
        #file_data = json.load(fp)
        
        json.dump(file_data, fp)
        
     
    # for plotting
    DiceSimilarities1,DiceSimilarities2,DiceSimilarities3,DiceSimilarities4,DiceSimilarities5,CosineSimilarities1,CosineSimilarities2,CosineSimilarities3,CosineSimilarities4,CosineSimilarities5,diceS1,diceS2,diceS3,diceS4,diceS5=ReadJsonFiles(mainPathTest,Dataset,Attacks,returnDetails=True)
    #pdb.set_trace()
    
    DiceSimilarities1=EstimateNewValues(DiceSimilarities1,a_dice,b_dice)
    DiceSimilarities2=EstimateNewValues(DiceSimilarities2,a_dice,b_dice)
    DiceSimilarities3=EstimateNewValues(DiceSimilarities3,a_dice,b_dice)
    DiceSimilarities4=EstimateNewValues(DiceSimilarities4,a_dice,b_dice)
    DiceSimilarities5=EstimateNewValues(DiceSimilarities5,a_dice,b_dice)
    
    
    CosineSimilarities1=EstimateNewValues(CosineSimilarities1,a_cos,b_cos)
    CosineSimilarities2=EstimateNewValues(CosineSimilarities2,a_cos,b_cos)
    CosineSimilarities3=EstimateNewValues(CosineSimilarities3,a_cos,b_cos)
    CosineSimilarities4=EstimateNewValues(CosineSimilarities4,a_cos,b_cos)
    CosineSimilarities5=EstimateNewValues(CosineSimilarities5,a_cos,b_cos)
    

    PlotCorrelation([CosineSimilarities1,CosineSimilarities2,CosineSimilarities3,CosineSimilarities4,CosineSimilarities5],[diceS1,diceS2,diceS3,diceS4,diceS5],['DicePerMetric','Cosine_corrected'],Dataset,mainPath,CalculationType='mean',KeepNegatives=False)
    PlotCorrelation([DiceSimilarities1,DiceSimilarities2,DiceSimilarities3,DiceSimilarities4,DiceSimilarities5],[diceS1,diceS2,diceS3,diceS4,diceS5],['DicePerMetric','DiceSim_corrected'],Dataset,mainPath,CalculationType='mean',KeepNegatives=False) 

    
    



def QualityEstimation(Dataset,mainPath,Attacks=['Rician'],ignoreZero=False): 
    
    
    diceS,jaccardS,CosineSimilarities,DiceSimilarities=ReadJsonFiles(mainPath,Dataset,Attacks)
    DiceSimilarities1,DiceSimilarities2,DiceSimilarities3,DiceSimilarities4, DiceSimilarities5,CosineSimilarities1,CosineSimilarities2,CosineSimilarities3,CosineSimilarities4,CosineSimilarities5,diceS1,diceS2,diceS3,diceS4,diceS5=ReadJsonFiles(mainPath,Dataset,Attacks,returnDetails=True)
    

    #########################33
    # ignore zero similarities
    if ignoreZero:
        indCos=np.where(np.mean(CosineSimilarities,axis=(1,2))>0.05)[0]
        indDice=np.where(np.mean(DiceSimilarities,axis=(1,2))>0.05)[0]

        a_cos,b_cos=RegressionModel(CosineSimilarities[indCos,:,:],diceS[indCos])
        a_dice,b_dice=RegressionModel(DiceSimilarities[indDice,:,:],diceS[indDice])
    else:
        a_cos,b_cos=RegressionModel(CosineSimilarities,diceS)
        a_dice,b_dice=RegressionModel(DiceSimilarities,diceS)    
        
        
    ###########################3
    
    a_cos,b_cos=a_cos.item(),b_cos.item()
    a_dice,b_dice=a_dice.item(),b_dice.item()
    #pdb.set_trace()
    

    import scipy
    from scipy import stats

    PerformMetric=np.array(diceS) # diceS
    print('pearson correlation_min',stats.pearsonr(np.min(CosineSimilarities,axis=(1,2)),PerformMetric))

    print('pearson correlation_max',stats.pearsonr(np.max(CosineSimilarities,axis=(1,2)),PerformMetric))

    #pdb.set_trace()
    if ignoreZero:
        MeanCosineCorr=stats.pearsonr(np.mean(CosineSimilarities,axis=(1,2))[indCos],PerformMetric[indCos])
        print('pearson correlation_mean',stats.pearsonr(np.mean(CosineSimilarities,axis=(1,2))[indCos],PerformMetric[indCos]))
    else:
        MeanCosineCorr=stats.pearsonr(np.mean(CosineSimilarities,axis=(1,2)),PerformMetric)
        print('pearson correlation_mean',stats.pearsonr(np.mean(CosineSimilarities,axis=(1,2)),PerformMetric))


    print('pearson correlation_Dicemin',stats.pearsonr(np.min(DiceSimilarities,axis=(1,2)),PerformMetric))
    print('pearson correlation_Dicemax',stats.pearsonr(np.max(DiceSimilarities,axis=(1,2)),PerformMetric))
    if ignoreZero:
        MeanDiceCorr=stats.pearsonr(np.mean(DiceSimilarities,axis=(1,2))[indDice],PerformMetric[indDice])

        print('pearson correlation_Dicemean',stats.pearsonr(np.mean(DiceSimilarities,axis=(1,2))[indDice],PerformMetric[indDice]))
        
    else:
        MeanDiceCorr=stats.pearsonr(np.mean(DiceSimilarities,axis=(1,2)),PerformMetric)
        print('pearson correlation_Dicemean',stats.pearsonr(np.mean(DiceSimilarities,axis=(1,2)),PerformMetric))

    #pdb.set_trace()
    Done=0
    try:
        print('pearson correlation_Entropy_Dice',stats.pearsonr(1-Entropies,PerformMetric))
        Done=1
    except:
        print('error')
    print('minCosSim,dice',CalculateMAE_AUC(diceS,np.min(CosineSimilarities,axis=(1,2))))
    print('maxCosSim,dice',CalculateMAE_AUC(diceS,np.max(CosineSimilarities,axis=(1,2))))
    print('meanCosSim,dice',CalculateMAE_AUC(diceS,np.mean(CosineSimilarities,axis=(1,2))))    

    
    print('minDiceSim,dice',CalculateMAE_AUC(diceS,np.min(DiceSimilarities,axis=(1,2))))
    print('maxDiceSim,dice',CalculateMAE_AUC(diceS,np.max(DiceSimilarities,axis=(1,2))))
    print('meanDiceSim,dice',CalculateMAE_AUC(diceS,np.mean(DiceSimilarities,axis=(1,2))))

    

        
########################
       
        
    data={"cosineMinCorrelation":stats.pearsonr(np.min(CosineSimilarities,axis=(1,2)),PerformMetric), 
       "cosineMaxCorrelation":stats.pearsonr(np.max(CosineSimilarities,axis=(1,2)),PerformMetric),  
        "cosineMeanCorrelation":MeanCosineCorr,

          "DiceMinCorrelation":stats.pearsonr(np.min(DiceSimilarities,axis=(1,2)),PerformMetric),
          "DiceMaxCorrelation":stats.pearsonr(np.max(DiceSimilarities,axis=(1,2)),PerformMetric),
          "DiceMeanCorrelation":MeanDiceCorr,

          
         'minCosSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.min(CosineSimilarities,axis=(1,2))),
         'maxCosSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.max(CosineSimilarities,axis=(1,2))),
         'meanCosSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.mean(CosineSimilarities,axis=(1,2))),


         'minDiceSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.min(DiceSimilarities,axis=(1,2))),
         'maxDiceSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.max(DiceSimilarities,axis=(1,2))),
         'meanDiceSimdice_MaeClassificationAccuracy':CalculateMAE_AUC(diceS,np.mean(DiceSimilarities,axis=(1,2))),


            
          'Dice_RegressionCoefficents':[a_dice,b_dice],'Cosine_RegressionCoefficents':[a_cos,b_cos]
         }




        
    with open(os.path.join(mainPath, 'QualityEstimation.json'), 'w') as fp:
        json.dump(data, fp)   
        
    RegressionCoefCos=[a_cos,b_cos]
    RegressionCoefDice=[a_dice,b_dice]
    RegressionCoef=[]
    

    PlotCorrelation([CosineSimilarities1,CosineSimilarities2,CosineSimilarities3,CosineSimilarities4,CosineSimilarities5],[diceS1,diceS2,diceS3,diceS4,diceS5],['DicePerMetric','Cosine'],Dataset,mainPath,CalculationType='mean',RegressionCoef=RegressionCoefCos)  
        

    PlotCorrelation([DiceSimilarities1,DiceSimilarities2,DiceSimilarities3,DiceSimilarities4,DiceSimilarities5],[diceS1,diceS2,diceS3,diceS4,diceS5],['DicePerMetric','DiceSim'],Dataset,mainPath,CalculationType='mean',RegressionCoef=RegressionCoefDice) 
    
    





    
def PlotCorrelation(CosineSimilarities,jaccardS,ParameterNames,Dataset,mainPath,CalculationType='min',KeepNegatives=True,RegressionCoef=[]): 
    fig,ax1 = plt.subplots(figsize=(6,4)) # 6,4 for 2013
    print(Dataset)
    if Dataset=='DRIVE' or Dataset=='STARE' or Dataset=='IOSTAR':
        print('Plot for ', ParameterNames)
        #fig = plt.figure()

        # call the figure object's add_subplot() method to divide the current canvas. 
        #fig,ax = plt.subplots()
        # plot a figure with the green color.
        if len(CosineSimilarities[0])>0:
            print('0')


            if KeepNegatives:
                ax1.plot(np.mean(np.array(CosineSimilarities[0]),axis=(1,2)),np.array(jaccardS[0]),'+r',markersize=12)
            
            
            else:
                #pdb.set_trace()
                meanV=np.mean(np.array(CosineSimilarities[0]),axis=(1,2))

                ind=np.where(meanV<0)[0]
                ind=list(set(np.arange(0,len(CosineSimilarities[0]),1))-set(ind))
                ax1.plot(np.mean(np.array(CosineSimilarities[0]),axis=(1,2))[ind],np.array(jaccardS[0])[ind],'+r',markersize=12)  
                   
                    

        if len(CosineSimilarities[1])>0:
            print('1')
            if KeepNegatives:
                #pdb.set_trace()
                
                ax1.plot(np.mean(np.array(CosineSimilarities[1]),axis=(1,2)),np.array(jaccardS[1]),'+k',markersize=12)

            else:
                meanV=np.mean(np.array(CosineSimilarities[1]),axis=(1,2))
                
                ind=np.where(meanV<0)[0]
                ind=list(set(np.arange(0,len(CosineSimilarities[1]),1))-set(ind))
                ax1.plot(np.mean(np.array(CosineSimilarities[1]),axis=(1,2))[ind],np.array(jaccardS[1])[ind],'+k',markersize=12)            

        if len(CosineSimilarities[2])>0:
            print('2')
            #ax1.plot(np.min(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)
            if KeepNegatives:
                ax1.plot(np.mean(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'+b',markersize=12)
               
                
            else:
                meanV=np.mean(np.array(CosineSimilarities[2]),axis=(1,2))

                ind=np.where(meanV<0)[0]
                ind=list(set(np.arange(0,len(CosineSimilarities[2]),1))-set(ind))
                ax1.plot(np.mean(np.array(CosineSimilarities[2]),axis=(1,2))[ind],np.array(jaccardS[2])[ind],'+b',markersize=12)                  

        if len(CosineSimilarities[3])>0:
            print('3')
            #ax1.plot(np.min(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)
            if KeepNegatives:
                ax1.plot(np.mean(np.array(CosineSimilarities[3]),axis=(1,2)),np.array(jaccardS[3]),'+y',markersize=12)
               
                
            else:
                meanV=np.mean(np.array(CosineSimilarities[3]),axis=(1,2))

                ind=np.where(meanV<0)[0]
                ind=list(set(np.arange(0,len(CosineSimilarities[3]),1))-set(ind))
                ax1.plot(np.mean(np.array(CosineSimilarities[3]),axis=(1,2))[ind],np.array(jaccardS[3])[ind],'+y',markersize=12)         
        if len(CosineSimilarities)>4:         
            if len(CosineSimilarities[4])>0:
                print('4')
                #ax1.plot(np.min(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)
                if KeepNegatives:
                    ax1.plot(np.mean(np.array(CosineSimilarities[4]),axis=(1,2)),np.array(jaccardS[4]),'+g',markersize=12)


                else:
                    meanV=np.mean(np.array(CosineSimilarities[4]),axis=(1,2))

                    ind=np.where(meanV<0)[0]
                    ind=list(set(np.arange(0,len(CosineSimilarities[4]),1))-set(ind))
                    ax1.plot(np.mean(np.array(CosineSimilarities[4]),axis=(1,2))[ind],np.array(jaccardS[4])[ind],'+g',markersize=12)                              
        #ax.grid(False)
        #ax.set_facecolor('white')
        ax1.set_xticks(np.arange(0.0,1.1,0.1))
        ax1.set_yticks(np.arange(0.0,1.1,0.1))     
    else:
        

        
        if CalculationType=='mean':
            #pdb.set_trace()
            if len(CosineSimilarities[0])>0:
                print('0')
                #ax1.plot(np.mean(np.array(CosineSimilarities[0]),axis=(1,2)),jaccardS[0],'*r',markersize=12)
                if KeepNegatives:
                    ax1.plot(np.mean(np.array(CosineSimilarities[0]),axis=(1,2)),np.array(jaccardS[0]),'*r',markersize=12)

                else:
                    meanV=np.mean(np.array(CosineSimilarities[0]),axis=(1,2))

                    ind=np.where(meanV<0)[0]
                    ind=list(set(np.arange(0,len(CosineSimilarities[0]),1))-set(ind))
                    ax1.plot(np.mean(np.array(CosineSimilarities[0]),axis=(1,2))[ind],np.array(jaccardS[0])[ind],'*r',markersize=12)                
               
                
                
                
            if len(CosineSimilarities[1])>0:
                print('1')
                if KeepNegatives:
                    ax1.plot(np.mean(np.array(CosineSimilarities[1]),axis=(1,2)),np.array(jaccardS[1]),'*k',markersize=12)

                    #pdb.set_trace() 
                else:
                    meanV=np.mean(np.array(CosineSimilarities[1]),axis=(1,2))

                    ind=np.where(meanV<0)[0]
                    ind=list(set(np.arange(0,len(CosineSimilarities[1]),1))-set(ind))
                    ax1.plot(np.mean(np.array(CosineSimilarities[1]),axis=(1,2))[ind],np.array(jaccardS[1])[ind],'*k',markersize=12)            
               
            if len(CosineSimilarities[2])>0:
                print('2')
                #ax1.plot(np.min(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)
                if KeepNegatives:
                    ax1.plot(np.mean(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)

                else:
                    meanV=np.mean(np.array(CosineSimilarities[2]),axis=(1,2))

                    ind=np.where(meanV<0)[0]
                    ind=list(set(np.arange(0,len(CosineSimilarities[2]),1))-set(ind))
                    ax1.plot(np.mean(np.array(CosineSimilarities[2]),axis=(1,2))[ind],np.array(jaccardS[2])[ind],'*b',markersize=12)   
                    
                    
            if len(CosineSimilarities[3])>0:
                print('3')
                #ax1.plot(np.min(np.array(CosineSimilarities[2]),axis=(1,2)),np.array(jaccardS[2]),'*b',markersize=12)
                if KeepNegatives:
                    ax1.plot(np.mean(np.array(CosineSimilarities[3]),axis=(1,2)),np.array(jaccardS[3]),'*y',markersize=12)

                else:
                    meanV=np.mean(np.array(CosineSimilarities[3]),axis=(1,2))

                    ind=np.where(meanV<0)[0]
                    ind=list(set(np.arange(0,len(CosineSimilarities[3]),1))-set(ind))
                    ax1.plot(np.mean(np.array(CosineSimilarities[3]),axis=(1,2))[ind],np.array(jaccardS[3])[ind],'*y',markersize=12)  

                    
            ax1.set_xticks(np.arange(0.0,1.1,0.1))
            ax1.set_yticks(np.arange(0.0,1.1,0.1)) 


            #ax1.set_xticklabels(np.arange(0.0,1.1,0.1),fontsize=12)
            #ax1.set_yticklabels(np.arange(0.0,1.1,0.1),fontsize=12)          

            
    if len(RegressionCoef)>0:
        x=np.arange(0,0.99,0.01)#1.00001,0.01)
        a,b=RegressionCoef
        y=a*x+b
        ind=np.where(y<0)[0]
        ind2=np.where(y>0.99)[0]
        
        ind=list(set(np.arange(0,len(x),1))-set(ind)-set(ind2))            
        ax1.plot(x[ind],y[ind],markersize=12) 
        
        
        ax1.set_xticks(np.arange(0.0,1.1,0.1))
        ax1.set_yticks(np.arange(0.0,1.1,0.1)) 
        
    #ax1.set_xticks(np.arange(0.0,1.6,0.1))
    #ax1.set_yticks(np.arange(0.0,1.1,0.1))    
    #pdb.set_trace()
    plt.savefig(os.path.join(mainPath,Dataset+'_'+CalculationType+ParameterNames[1]+ParameterNames[0]+'.png'),dpi=300)
    
        
def CalculateMAE_AUC(diceS,CosineSimilaritiesMinS):
    Mae=np.mean(np.abs(np.array(diceS) - np.array(CosineSimilaritiesMinS)))
    from sklearn.metrics import auc,roc_curve
    diceS_binary=np.uint8(np.array(diceS)>0.6999)

    fpr, tpr, thresholds = roc_curve(diceS_binary, np.array(CosineSimilaritiesMinS), pos_label=1)
    #print(Mae, auc(fpr, tpr))
    return Mae, auc(fpr, tpr)
        
        
        
def FlattenList(CosineSimilarities,DiceSimilarities):

    CosineSimilaritiesList=np.array(CosineSimilarities.copy())
    DiceSimilaritiesList=np.array(DiceSimilarities.copy())

    CosineSimilarities=[]
    DiceSimilarities=[]

    for i in range(len(CosineSimilaritiesList)):
        #pdb.set_trace()
        for t in range(len(CosineSimilaritiesList[i,0])):
            a=CosineSimilaritiesList[i,0][t]
            #a=[a[0],a[2],a[3],a[5]]
            a=[a[0],a[1],a[2]]
            
            CosineSimilarities.append([a])
            
            a=DiceSimilaritiesList[i,0][t]
            #a=[a[0],a[2],a[3],a[5]]
            a=[a[0],a[1],a[2]]
            
            DiceSimilarities.append([a])





    CosineSimilarities=np.array(CosineSimilarities)
    DiceSimilarities=np.array(DiceSimilarities)
    return CosineSimilarities,DiceSimilarities

