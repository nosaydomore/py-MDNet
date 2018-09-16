#-*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# net = Net()
#
# #create your optimizer
# for i in range(1):
#     optimizer = optim.SGD(net.parameters(), lr = 0.01)
#     input = Variable(torch.randn(1, 1, 32, 32))
#     # optimizer.zero_grad()
#     # print 'conv1.bias.grad',net.conv1.bias.grad
#     output = net(input)
#     print output.shape
#     # criterion = nn.MSELoss()
#     # target = Variable(torch.range(1, 10).reshape(1, 10))
#     # loss = criterion(output, target)    # 损失
#     # print 'loss:',loss
#     # loss.backward() # 梯度
#     # print 'conv1.bias.grad',net.conv1.bias.grad
#     # optimizer.step()    # 更新
