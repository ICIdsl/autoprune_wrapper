import os
import sys
import copy
import random
import logging
import argparse
from collections import OrderedDict

import torch
import numpy as np

import utils 
import autoprune

def prune(pl, network):
    model, convsToIgnore= utils.getModelAndIgnoredConvs(network)
    prunedModel = autoprune.pruneNetwork(pl, model, network,\
                                         rankingType= 'l1-norm',\
                                         ignoreKws=convsToIgnore)
    print(prunedModel)
    uModelSize= utils.modelSize(model)
    pModelSize= utils.modelSize(prunedModel)
    print(f"Original model size= {uModelSize} MB")
    print(f"Pruned model size= {pModelSize} MB ({pModelSize/uModelSize:.2f}%)")

parser= argparse.ArgumentParser()
parser.add_argument('--pl', help='pruning level')
parser.add_argument('--net', help='network to prune')

args= parser.parse_args()
prune(float(args.pl), args.net)

