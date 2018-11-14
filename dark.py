#DARKNETS

from fastai.layers import *
from fastai.torch_core import *
from torch.autograd import Variable
# from pap import PositionalAveragePooling, PSEModule

# class GWApool(nn.Module):
#     '''GWA
#     https://arxiv.org/pdf/1809.08264.pdf'''
#     def __init__(self, k, classes):
#         super().__init__()
#         self.classes = classes
#         self.k = k
#         self.w = Variable(torch.ones(k), requires_grad=True)
#         self.b = Variable(torch.ones(1), requires_grad=True)
#         self.l = nn.Linear(k, classes)
        
#     def forward(self, x):
# #         n_features = np.prod(x.size()[1:])
# #         x = x.view(-1, n_features)
#         x = x.float()
#         M = torch.exp(self.w.view(1, self.k, 1,1) * x + self.b).sigmoid()
#         return self.l((x * M / M.sum()).sum(dim=[-2,-1]))
        

class ResLayer(nn.Module):
    "Resnet style `ResLayer`"
    def __init__(self, ni:int):
        "create ResLayer with `ni` inputs"
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)

    def forward(self, x): return x + self.conv2(self.conv1(x))

    

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class SEModule(nn.Module):
    def __init__(self, ch, re=16):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid()
                               )
    def forward(self, x):
        return x * self.se(x)
    
class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = SEModule(ch, re)
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
        
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
class ResLayerSE(nn.Module):
    "Resnet style `ResLayer`"
    def __init__(self, ni:int):
        "create ResLayer with `ni` inputs"
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)
        
        #SE block
        self.SE = SEModule(ni, 2)
        
    def forward(self, x): 
        return x + self.SE(self.conv2(self.conv1(x)))
 
    
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