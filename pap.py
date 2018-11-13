from fastai.torch_core import *

class PositionalAveragePooling(nn.Module):
    def __init__(self, kernel_size=1, stride=None, padding=0, ceil_mode=False,  count_include_pad=True):
        super(PositionalAveragePooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

    def forward(self, x):
        dim = x.shape[1]
        print(dim)
        filter = F.avg_pool2d(x, self.kernel_size, self.stride,
                                  self.padding, self.ceil_mode, self.count_include_pad)
        print(x.shape)
        x = torch.sum(x, 1, keepdim=True)/dim
        return x   
  
class PSEModule(nn.Module):

    def __init__(self, channels):
        super(PSEModule, self).__init__()
        self.pap = PositionalAveragePooling()
        self.conv1 = nn.Conv2d(1, channels // 2, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.pap(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x