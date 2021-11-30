import torchvision
import numpy as np

def getModelAndIgnoredConvs(name):
    print(f"Loading model {name}")
    if name.startswith('resnet'):
        assert '_' in name, "Resnets need to be specified with depth as resnet_{depth}"
        from models.resnet import resnet
        depth = int(name.split('_')[1])
        return resnet(depth, 1000, pretrained=True), ['is::conv1', 'contains::downsample']
    elif name == 'gated_resnet_50':
        from models.gated_resnet import gated_resnet
        depth = int(name.split('_')[2])
        return gated_resnet(depth, 1000, pretrained=True), ['is::conv1', 'contains::downsample'] 
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


