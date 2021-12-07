import os
import sys
import copy
import random
import logging
import argparse
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

import utils 
import autoprune
import autoprune.model_graph as graphCreator
from autoprune.graph_nodes import Conv2DNode

class GatedConv2DNode(Conv2DNode):
    def prune(self, seenNodes):
        if self in seenNodes:
            return
        seenNodes.append(self)
        
        if ((self.module.in_channels/self.module.groups) != self.module.weight.shape[1]): 
            self.module.weight= torch.nn.Parameter(\
                    autoprune.utils.reshapeTensor(self.module.weight, self.inChannels, 1))
        
        if (self.module.out_channels != self.module.weight.shape[0]):
            self.module.weight= torch.nn.Parameter(\
                    autoprune.utils.reshapeTensor(self.module.weight, self.filters, 0))
            
            if hasattr(self, 'gateModule'): 
                self.gateModule.ipFeats= len(self.filters) 
                self.gateModule.opFeats= len(self.filters) 
                self.gateModule.weight= torch.nn.Parameter(torch.ones(len(self.filters)))
            
            if self.module.bias is not None:
                self.module.bias= torch.nn.Parameter(\
                        autoprune.utils.reshapeTensor(self.module.bias, self.filters, 0))
        
        for nextNode in self.nextNodes:
            nextNode.prune(seenNodes)

def getModelGateModules(model, gateModule, ignoreKws):
    prevConv= None
    gateModules= {}
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prevConv= n
        if isinstance(m, gateModule):
            if prevConv is None:
                raise ValueError(f"Input model has gate module {n} before any conv module!")
            elif autoprune.utils.layerToIgnore(prevConv, ignoreKws):
                print(f"WARNING: Conv {prevConv} is not being pruned but has gate layer {n}")
            else:
                gateModules[prevConv] = (n,m)
    return gateModules 

def createLayerNode(name, module, node):
    if isinstance(module, torch.nn.Conv2d):
        return GatedConv2DNode(name, module, node)
    elif isinstance(module, torch.nn.BatchNorm2d):
        return autoprune.graph_nodes.BatchNorm2DNode(name, module, node) 
    elif isinstance(module, torch.nn.Linear):
        return autoprune.graph_nodes.LinearNode(name, module, node) 
    else:
        raise NotImplementedError(f"Handling {module} not implemented")

def buildLayerGraph(model, execGraph, gateModules):
    modules= graphCreator.baseModules(model) 
    translations= graphCreator.getExecGraphToLayerNameTranslations(modules, model, execGraph)
    print(f"Creating Graph")
    createdNodes= []
    root= graphCreator.getRootNode(model, translations, createdNodes)
    graphCreator.createGraph(root, root.node, translations, createdNodes) 
    return root

def updateConvNodesWithGateModules(node, modelGateModules, seenNodes):
    if node in seenNodes:
        return
    if node.name in modelGateModules.keys():
        node.gateModule= modelGateModules[node.name][1]
        node.gateModuleName= modelGateModules[node.name][0]

    seenNodes.append(node)
    for nextNode in node.nextNodes:
        updateConvNodesWithGateModules(nextNode, modelGateModules, seenNodes)

def getImageNetDataset():
    from imagenet import ImageNet
    dataset= ImageNet('/idslF/data/imagenet')
    dataset.setupDataLoaders()
    return dataset

def subSampleDataset(dataset, numMb=30, batchSize=64):
    allIndices = list(range(len(dataset)))
    chosenIndexes = np.random.choice(allIndices, numMb*batchSize)
    subSampler = torch.utils.data.sampler.SubsetRandomSampler(chosenIndexes)
    subLoader = torch.utils.data.DataLoader(
        dataset,
        sampler=subSampler,
        batch_size=batchSize,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return subLoader

def getTaylorSensitivities(model, dataLoader, gateLayer, gpu=0):             
    device = f"cuda:{gpu}"
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device, non_blocking=True) 
    model.eval()
    model.zero_grad()
    with tqdm(total=len(dataLoader), desc='Gradient Computation') as iterator:
        for batchIdx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.cuda(device, non_blocking=True),\
                                targets.cuda(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            iterator.set_postfix({'loss': loss.item()})
            iterator.update(1)
        
    model.to('cpu')
    sensitivities = {}
    for n,m in model.named_modules():
        if isinstance(m, gateLayer):
            m.weight.grad /= len(dataLoader)
            metric = m.weight.grad.data.pow(2).view(m.weight.size(0), -1).sum(dim=1)
            sensitivities[n] = metric

    return sensitivities

def rankFilters(model, ignoreKws, gateModuleInst, allGateModules, gpu=0):
    dataset = getImageNetDataset()
    subsetLoader= subSampleDataset(dataset.trainDataset)
    sensitivities= getTaylorSensitivities(copy.deepcopy(model), subsetLoader, gateModuleInst, gpu) 
    
    ranking= []
    for gateLayer, sens in sensitivities.items():
        conv= [k for k,v in allGateModules.items() if v[0] == gateLayer][0]
        if autoprune.layerToIgnore(conv, ignoreKws):
            continue
        ranking += [(conv, i, s.item()) for i,s in enumerate(sens)]
    return sorted(ranking, key=lambda i: i[2])

def checkPruning(model, gateModule):
    logging.info(f"Checking prunining process")
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.out_channels == m.weight.shape[0], f"Layer {n} pruned incorrectly"
            assert (m.in_channels // m.groups) == m.weight.shape[1],\
                                                                f"Layer {n} pruned incorrectly"
        elif isinstance(m, gateModule):
            assert (m.ipFeats == m.opFeats) and (m.ipFeats == m.weight.shape[0]),\
                    f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.BatchNorm2d):
            assert m.num_features == m.weight.shape[0], f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.Linear):
            try:
                assert m.in_features == m.weight.shape[1], f"Layer {n} pruned incorrectly"
            except Exception as e:
                breakpoint()

def pruneNetwork(pl,
                 model,
                 network,
                 gateModule,
                 ignoreKws=None,
                 minFiltersKept=2,
                 maintainNetworkWidth=True):
    
    assert 0 <= pl < 1, "Pruning level must be value in range [0,1)" 
    prunedModel= copy.deepcopy(model)
    execGraph = autoprune.getExecutionGraph(network, prunedModel)
    modelGateModules= getModelGateModules(prunedModel, gateModule, ignoreKws)
    root= buildLayerGraph(prunedModel, execGraph, modelGateModules) 
    updateConvNodesWithGateModules(root, modelGateModules, [])
    autoprune.runPrePruningPasses(root)
    autoprune.updatePruningLimits(root, pl, minFiltersKept, maintainNetworkWidth)
    globalRanking = rankFilters(prunedModel, ignoreKws, gateModule, modelGateModules, gpu=0)
    autoprune.identifyFiltersToPrune(root, globalRanking, prunedModel, pl)
    root.prune([])
    autoprune.checkPruning(prunedModel)
    return prunedModel

def registerCustomFunctions():
    autoprune.utils.registerCustomFunction(graphCreator, 'createLayerNode', createLayerNode)

def prune(pl, network):
    registerCustomFunctions()
    (model, gateModule), convsToIgnore= utils.getGatedModelAndIgnoredConvs(network)
    prunedModel = pruneNetwork(pl, model, network, gateModule, ignoreKws=convsToIgnore)
    print(prunedModel)
    uModelSize= utils.modelSize(model)
    pModelSize= utils.modelSize(prunedModel)
    print(f"Original model size= {uModelSize} MB")
    print(f"Pruned model size= {pModelSize} MB ({(pModelSize/uModelSize) * 100:.2f}%)")

parser= argparse.ArgumentParser()
parser.add_argument('--pl', help='pruning level')
parser.add_argument('--net', help='network to prune')

args= parser.parse_args()
prune(float(args.pl), args.net)

