import torch
import torchvision
import numpy as np
from tqdm import tqdm

def getGatedModelAndIgnoredConvs(name):
    if name == 'gated_resnet_50':
        from models.gated_resnet import gated_resnet, GateLayer
        depth = int(name.split('_')[2])
        return (gated_resnet(depth, 1000, pretrained=True), GateLayer),\
                        ['is::conv1', 'contains::downsample'] 
    else:
        raise NotImplementedError(f"Loading for gated network {name} not implemented.")

def getModelAndIgnoredConvs(name):
    print(f"Loading model {name}")
    if name.startswith('resnet'):
        assert '_' in name, "Resnets need to be specified with depth as resnet_{depth}"
        from models.resnet import resnet
        depth = int(name.split('_')[1])
        return resnet(depth, 1000, pretrained=True), ['is::conv1', 'contains::downsample']
    elif name == 'mobilenetv2':
        from models.mobilenet import mobilenetv2
        return mobilenetv2(pretrained=True), ['is::conv1']
    elif name == 'mnasnet':
        from models.mnasnet import mnasnet
        return mnasnet(1000, pretrained=True), ['is::layers.0', 'is::layers.3', 'is::layers.6']
    elif name == 'squeezenet':
        from models.squeezenet import squeezenet
        return squeezenet(1000, pretrained=True), ['is::features.0']
    elif name == 'googlenet':
        from models.googlenet_noaux import googlenet 
        return googlenet(pretrained=False), ['is::conv1.conv', 'is::conv2.conv', 'is::conv3.conv']
    else:
        raise NotImplementedError(f"Loading for network {name} not implemented.")

def modelSize(model):
    return sum(np.prod(p.shape) for p in model.parameters()) * 4 / 1e6

class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target) : 
    batchSize = target.size(0) 
    # torch.topk returns the values and indices of the k(5) largest elements in dimension 1 in a sorted manner
    _, indices = output.topk(5, 1, True, True)
    indices.t_()
    correctPredictions = indices.eq(target.view(1,-1).expand_as(indices))

    res = [] 
    for k in (1,5) : 
        correctK = correctPredictions[:k].reshape(-1).float().sum(0)
        res.append(correctK.mul_(100.0 / batchSize))

    return res

def testNetwork(model, loader, criterion, gpuList, verbose=False) :  
    model.to('cuda:' + str(gpuList[0])) 
    model.eval()
        
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Inference', leave=verbose) as t:
            for batchIdx, (inputs, targets) in enumerate(loader): 
                device = 'cuda:' + str(gpuList[0])
                inputs, targets = inputs.cuda(device, non_blocking=True),\
                                  targets.cuda(device, non_blocking=True)
                
                outputs = model(inputs) 
                loss = criterion(outputs, targets)
                
                prec1, prec5 = accuracy(outputs.data, targets.data)
                losses.update(loss.item()) 
                top1.update(prec1.item()) 
                top5.update(prec5.item())

                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg
                })
                t.update(1)
    
    return (losses.avg, top1.avg, top5.avg)



