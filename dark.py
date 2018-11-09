#DARKNETS

from fastai.layers import *
from fastai.torch_core import *

class ResLayer(nn.Module):
    "Resnet style `ResLayer`"
    def __init__(self, ni:int):
        "create ResLayer with `ni` inputs"
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)

    def forward(self, x): return x + self.conv2(self.conv1(x))

class ResLayerSE(nn.Module):
    "Resnet style `ResLayer`"
    def __init__(self, ni:int):
        "create ResLayer with `ni` inputs"
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)
        #SE block
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                     nn.Conv2d(ni, ni//2,1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(ni//2, ni,1),
                     nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ni,ni,1),
                     nn.Sigmoid())
        
    def forward(self, x): 
        r = self.conv2(self.conv1(x))
        return x + torch.addcmul(r * self.cSE(r), 1, r, self.sSE(r))
 
    
class Darknet(nn.Module):
    "https://github.com/pjreddie/darknet"
    def make_group_layer(self, ch_in:int, num_blocks:int, stride:int=1, se=False):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        if se: res_layer = ResLayerSE
        else: res_layer = ResLayer
        return [conv_layer(ch_in, ch_in*2,stride=stride)
               ] + [(res_layer(ch_in*2)) for i in range(num_blocks)]
    
    def __init__(self, num_blocks:Collection[int], num_classes:int, nf=32, se=False):
        "create darknet with `nf` and `num_blocks` layers"
        super().__init__()
        layers = [conv_layer(4, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1), se=se)
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

def dark_small(): return Darknet([1,2,4,4,3], 28, 32)
def dark_53(): return Darknet([1,2,8,8,4], 28, 32)