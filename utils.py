import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torch.tensor
import tensorflow as tf



def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((1,1,kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 1)
    return out_filter


def gauss_kernel(kernlen=21, nsig=3, channels=3):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1,1,kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=1)
    return out_filter   # kernel_size=21

class SeparableConv2d(nn.Module):
    def __init__(self):
        super(SeparableConv2d, self).__init__()
        kernel = gauss_kernel(21, 3, 3)
        kernel = torch.FloatTensor(kernel).cuda()
        ## kernel_point = [[1.0]]
        ## kernel_point = torch.FloatTensor(kernel_point).unsqueeze(0).unsqueeze(0)
        # kernel = torch.FloatTensor(kernel).expand(3, 3, 21, 21)   # torch.expand(）向输入的维度前面进行扩充，输入为三通道时，将weight扩展为[3,3,21,21]
        ## kernel_point = torch.FloatTensor(kernel_point).expand(3,3,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        # self.pointwise = nn.Conv2d(1, 1, 1, 1, 0, 1, 1,bias=False)    # 单通道时in_channels=1，out_channels=1,三通道时，in_channels=3, out_channels=3  卷积核为随机的
        ## self.weight_point = nn.Parameter(data=kernel_point, requires_grad=False)

    def forward(self, img1):
        x = F.conv2d(img1, self.weight, groups=1,padding=10)
        ## x = F.conv2d(x, self.weight_point, groups=1, padding=0)  #卷积核为[1]
        # x = self.pointwise(x)
        return x





