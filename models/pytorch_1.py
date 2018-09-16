# -*- coding:utf-8 -*-
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# import torch
#
# x = Variable(torch.ones(2, 2), requires_grad=True)
# y = x + 2
# print y
# z = y * y *3
# print z
# out = z.mean()
# print out
# out.backward()
# print x.grad

# import torch
# from torch.autograd import Variable
# x = Variable(torch.randn(5, 5))
# y = Variable(torch.randn(5, 5))
# z = Variable(torch.randn(5, 5), requires_grad=True)
# a = x + y  # x, y的 requires_grad的标记都为false， 所以输出的变量requires_grad也为false
# print a.requires_grad
#
# b = a + z
# print b.requires_grad
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from learn_pytorch import Net
import torch.nn as nn
import torch.optim as optim
import torch

transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
train_dataset = dsets.CIFAR10(root = './data',train = True,
                              transform = transform,
                              download=False)
test_dataset = dsets.CIFAR10(root='./data',train=False,download=False,
                             transform = transform)
image, label = train_dataset[0]
print image.size()
print label

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10,
                          shuffle=True,
                          num_workers=2)
ohe = OneHotEncoder()
ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
# data_iter = iter(train_loader)
netclass = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(netclass.parameters(), lr=0.001, momentum=0.9)
recentLoss = 0.
for j in range(50):
    for i ,data in enumerate(train_loader,0):
    # images, labels = next(data_iter)
        images, labels = data
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
    # labels=labels.reshape(labels.shape[0],1)
    # labels=ohe.transform(labels).toarray()
    # labels = Variable(torch.Tensor(labels))
    # print labels
        optimizer.zero_grad()
        percision = netclass(images)
        loss = criterion(percision,labels)
        loss.backward()
        optimizer.step()
        recentLoss = recentLoss + loss
        if i %200 == 0:
            print 'epoch', j, 'recentLoss:', recentLoss / 200
            recentLoss = 0
torch.save(netclass.state_dict(), 'net_cla_params.pkl')
# for images, labels in train_loader:
#
#     pass
# print images.size()