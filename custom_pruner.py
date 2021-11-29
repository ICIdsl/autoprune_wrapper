import os
import sys
import copy
import random
import logging
from collections import OrderedDict

import torch
import numpy as np

import autoprune

def getModel(name):
    print(f"Loading model {name}")
    if name.startswith('resnet'):
        from models.resnet import resnet
        depth = int(name.split('_')[1])
        return resnet(depth, 1000, pretrained=True) 
    elif name == 'gated_resnet_50':
        from models.gated_resnet import gated_resnet
        depth = int(name.split('_')[2])
        return gated_resnet(depth, 1000, pretrained=True) 
    elif name == 'mobilenetv2':
        from models.mobilenet import mobilenetv2
        return mobilenetv2(pretrained=True)
    else:
        raise NotImplementedError(f"Loading for network {name} not implemented.")

def randomRanker(model, ignore=None):
    globalRanking = []
    perLayer = {} 

    # create global ranking
    layers = []
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            metric = np.random.random([m.out_channels])
            perLayer[n] = [(i, x) for i,x in enumerate(metric)]
            if not autoprune.utils.layerToIgnore(n, ignore):
                globalRanking += [(n, i, x) for i,x in enumerate(metric)]
    globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
    return perLayer, globalRanking
     
def customRanker():
    pl = 0.5
    model = getModel('resnet_50')
    convsToIgnore = ['is::conv1', 'contains::downsample']
    
    prunedModel, channelsKept = autoprune.pruneNetwork(pl, model,\
                                                       customRanker = randomRanker,\
                                                       ignoreKws=convsToIgnore)

customRanker()

