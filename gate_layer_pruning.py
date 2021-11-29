import os
import sys
import copy
import random
import logging

import torch
import numpy as np

import autoprune

def getModel(name):
    if name == 'resnet_50':
        from models.resnet import resnet
        depth = int(name.split('_')[1])
        return resnet(depth, 1000, pretrained=True) 
    elif name == 'gatedResnet_50':
        from models.gatedResnet import gatedResnet
        depth = int(name.split('_')[2])
        return gatedResnet(depth, 1000, pretrained=True) 

def gatedPerformPruning(prunedModel, filtersToKeep, connectivity):
    logging.info(f"Removing Filters")
    from models.gatedResnet import GateLayer
    modelDict = dict(prunedModel.named_modules())
    for n,m in prunedModel.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            opFilters = [x for x,y in filtersToKeep[n]]
            autoprune.utils.reshapeConvLayer(m, opFilters, ofm=True)
            for layer in connectivity[n]:
                module = modelDict[layer]
                if isinstance(module, torch.nn.Conv2d):
                    autoprune.utils.reshapeConvLayer(module, opFilters, ofm=False)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    autoprune.utils.reshapeBnLayer(module, opFilters)
                elif isinstance(module, torch.nn.Linear):
                    autoprune.utils.reshapeLinearLayer(module, m, opFilters)
        elif isinstance(m, GateLayer):
            m.ipFeats = len(opFilters)
            m.opFeats = len(opFilters)
            newWeight = autoprune.utils.reshapeTensor(m.weight, opFilters, axis=0)
            m.weight = torch.nn.Parameter(newWeight)
    autoprune.checkPruning(prunedModel)
    return prunedModel

def gatedPruneNetwork(pl,
                         model,
                         ignoreKws=None,
                         minFiltersKept=2,
                         customRanker=None,
                         rankingType='l1-norm',
                         maintainNetworkWidth=True):
    assert 0 <= pl < 1, "Pruning level must be value in range [0,1)" 
    
    dependencies = autoprune.getDependencies(model)
    connectivity, joinNodes, localDeps, globalDeps =\
                                    autoprune.categoriseDependencies(dependencies)
    
    channelsPerLayer, globalRanking = autoprune.rankFilters(rankingType, model, ignoreKws,\
                                                                customRanker)
    
    prnLimits = {k:minFiltersKept for k in channelsPerLayer.keys()}
    if maintainNetworkWidth:
        prnLimits = autoprune.limitGlobalDepsPruning(pl, prnLimits, channelsPerLayer,\
                                                         globalDeps)
    # After this step, "prunedModel" will only have the description parameters set correctly, but
    # not the weights themselves. "filtersToKeep" will have the channels to keep for each layer 
    prunedModel, filtersToKeep =\
                autoprune.identifyFiltersToPrune(pl, model, channelsPerLayer,\
                                                    globalRanking, connectivity,\
                                                    joinNodes, localDeps+globalDeps,\
                                                    prnLimits)
    prunedModel = gatedPerformPruning(prunedModel, filtersToKeep, connectivity)
    return prunedModel, filtersToKeep

def gatedNetworkPruning():
    pl = 0.5
    model = getModel('gatedResnet_50')
    convsToIgnore = ['is::conv1', 'contains::downsample']
    prunedModel, channelsKept = gatedPruneNetwork(pl, model,\
                                                      rankingType= 'l1-norm',\
                                                      ignoreKws=convsToIgnore)
    
gatedNetworkPruning()

