# Adapted from https://github.com/activatedgeek/LeNet-5

import torch.nn as nn

class Layer(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return self.func(input)

class C1(Layer):
    
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

class C2(Layer):
    
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        
class C3(Layer):
    
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )

class F4(Layer):
    
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

class F5(Layer):
    
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

class LeNet5(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, input):
        output = self.c1(input)
        output = self.c2_1(output) + self.c2_2(output)
        output = self.c3(output)
        output = output.squeeze()
        output = self.f4(output)
        return self.f5(output)
