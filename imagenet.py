import os
import sys
import copy
import random
import logging

import torch
import torchvision
import numpy as np

class ImageNet():
    def __init__(self, dataLoc, ndw=4, testBs=32, trainBs=32): 
        self.loc= dataLoc
        self.numDataWorkers= ndw
        self.testBatchSize= testBs
        self.trainBatchSize= trainBs
        
    def setupTransforms(self):
        self.trainTransform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        self.testTransform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

    def setupDatasets(self):
        trainRoot = os.path.join(self.loc, 'train')
        self.trainDataset = torchvision.datasets.ImageFolder(root= trainRoot,\
                                                          transform= self.trainTransform)
        
        testRoot = os.path.join(self.loc, 'validation')
        self.testDataset = torchvision.datasets.ImageFolder(root= testRoot,\
                                                        transform= self.testTransform)
    
    def setupLoaders(self):
        self.trainLoader = torch.utils.data.DataLoader(self.trainDataset,\
                                                       batch_size= self.trainBatchSize,\
                                                       shuffle= True,\
                                                       drop_last= False,\
                                                       num_workers= self.numDataWorkers)

        self.testLoader = torch.utils.data.DataLoader(self.testDataset,\
                                                      batch_size= self.testBatchSize,\
                                                      shuffle= False,\
                                                      drop_last= False,\
                                                      num_workers= self.numDataWorkers)
        
    def setupDataLoaders(self):
        self.setupTransforms()
        self.setupDatasets()
        self.setupLoaders()

            


