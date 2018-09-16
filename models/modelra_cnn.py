import torch.nn as nn
import os
import scipy.io
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import copy

# class AttentionCrop(nn.Module):
#     def __init__(self):
#         super(AttentionCrop,self).__init__()
#     def forward(self,x):
def AttentionCrop(x,image):
    

class RA_CNN(nn.Module):
    def __init__(self):
        super(RA_CNN,self).__init__()
        self.scale_1 = nn.Sequential(
            OrderedDict([('conv1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv2', nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv3', nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv4', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv5', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))
                         ]))
        self.APN_1 = nn.Sequential(OrderedDict([('apn_pool', nn.MaxPool2d(kernel_size=2, stride=2)),
                                                ('get_abc1', nn.Linear(1024, 1024), ('tanh', nn.Tanh()),
                                                ('get_abc2', nn.Linear(1024, 3)),
                                                ('sigmoid', nn.Sigmoid()))]))

        self.scale_2 = nn.Sequential(
            OrderedDict([('conv1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv2', nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv3', nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv4', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv5', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))
                         ]))
        self.APN_2 = nn.Sequential(OrderedDict([('apn_pool', nn.MaxPool2d(kernel_size=2, stride=2)), (
            'get_abc1', nn.Linear(1024, 1024), ('tanh', nn.Tanh()), ('get_abc2', nn.Linear(1024, 3)),
            ('sigmoid', nn.Sigmoid()))]))

        self.scale_3 = nn.Sequential(
            OrderedDict([('conv1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv2', nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv3', nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                                           padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv4', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                         ('conv5', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))
                         ]))

    def forward(self,x):
        for name, module in self.scale_1.named_children():
            x = module(x)
        scale1 = copy.deepcopy(x)
        for name, module in self.APN_1.named_children():
            x = module(x)
        S2crop = AttentionCrop()
        for name, module in self.scale_2.named_children()
        for name,module in self.scale_2.named_children():

