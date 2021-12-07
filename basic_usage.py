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

def prune(pl, network, metric):
    model, convsToIgnore= utils.getModelAndIgnoredConvs(network)
    prunedModel = autoprune.pruneNetwork(pl, model, network,\
                                         rankingType= metric,\
                                         ignoreKws=convsToIgnore)
    print(prunedModel)
    uModelSize= utils.modelSize(model)
    pModelSize= utils.modelSize(prunedModel)
    print(f"Original model size= {uModelSize} MB")
    print(f"Pruned model size= {pModelSize} MB ({pModelSize/uModelSize:.2f}%)")

parser= argparse.ArgumentParser()
parser.add_argument('--pl', help='pruning level')
parser.add_argument('--net', help='network to prune')
parser.add_argument('--metric', help="supported pruning metrics: [l1-norm, taylor-fo]")

args= parser.parse_args()
prune(float(args.pl), args.net, args.metric)

