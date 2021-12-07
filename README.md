Autoprune
=
This is a wrapper repo that shows how to use the 
[autoprune library](https://github.com/ICIdsl/autoprune) which uses PyTorch JIT
Tracing functionality to extract an execution graph of a network to then automatically detect
dependencies between layers in order to prune dependent layers correctly. As it uses 
**torch.jit.trace** functionality, the models passed in must be written to be traceable.

Most things you want to do can be done without editing the **autoprune** library itself, and so
before running anything here run the command `git submodule update --init` to point to the latest
version of the library.

Currently, the models (connectivity) type supported by autoprune are:
    - No special connectivity : AlexNet, VGG, etc.
    - Residual connectivity : ResNets etc.
    - Depthwise convolution connectivity : MobileNets etc.
    - Concatenataion connectivity : SqueezeNets, GoogleNets etc.

The file **basic_usage.py** shows how to use **autoprune** without any modifications for the 
supported *l1-norm* pruning metric. 

Eg: To l1-norm prune resnet_18 by 20%, run 
```
python basic_usage.py --pl 0.2 --net resnet_18 --metric l1-norm
```

The file **taylor_pruning.py** shows how to add custom functionality to the library without 
modifying the library itself to implement the more complex *Taylor-first-order* pruning method.

Eg: To taylor prune resnet_50 by 20%, run 
```
python basic_usage.py --pl 0.2 --net gated_resnet_50
```

If you require to make even more detailed changes, you can change the library itself as this is
also open-sourced.

