#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 23:05:06 2019

@author: jayan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision
import torchvision.transforms as transforms
import math
import sys
import time
import os
from livelossplot import PlotLosses

lrt           = 0.001
start_epoch  = 1
num_epochs   = 200
batch_size   = 128
best_acc     = 0
criterion = nn.CrossEntropyLoss()

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./train_data', transform=transform_train, train=True, download=True)
test_dataset  = torchvision.datasets.CIFAR10(root='./test_data', transform=transform_test, train=False, download=True)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=80, num_workers=8, shuffle=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
            
        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2   = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5   = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                            nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output += F.relu(self.shortcut(input))
        output = F.relu(output)
        return output
    
class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64
        
        self.conv1  = nn.Conv2d(3, int(width_x * self.in_channels), 3, 1, 1, bias=True)     # For Cifar10
        #self.conv1  = nn.Conv2d(3, int(width_x * self.in_channels), 3, 2, 1, bias=True)     # For Tiny-ImageNet
        self.bn1    = nn.BatchNorm2d(int(width_x * self.in_channels))
        self.stage1 = self._make_layer(blocks[0], width_x, 32, 1)
        self.stage2 = self._make_layer(blocks[1], width_x, 64, 2)
        self.stage3 = self._make_layer(blocks[2], width_x, 128, 2)
        self.stage4 = self._make_layer(blocks[3], width_x, 256, 2)
        self.conv2  = nn.Conv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias=True)
        self.bn2    = nn.BatchNorm2d(int(width_x * 128))
        self.linear = nn.Linear(int(width_x * 128), num_classes)
        
    def _make_layer(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers  = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output
    
     
def SqNxt_23_1x(num_classes):
    return SqueezeNext(1.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_1x_v5(num_classes):
    return SqueezeNext(1.0, [2, 4, 14, 1], num_classes)

def SqNxt_23_2x(num_classes):
    return SqueezeNext(2.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_2x_v5(num_classes):
    return SqueezeNext(2.0, [2, 4, 14, 1], num_classes)

     

net = SqNxt_23_1x(10)
#net.apply(conv_init)
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
tmp = torch.randn(1, 3, 32, 32)
y   = net(tmp)
print(y, type(y), y.size())

def train(epoch):
    global train_loss
    global train_correct
    global train_total
    global optimizer
    net.train()
    train_loss = 0
    train_correct    = 0
    train_total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lrt, momentum=0.9, weight_decay=2e-4)
    
    print('Sqnet_1x_v1.1_elu Training Epoch: #%d, LR: %.4f'%(epoch, lrt))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        train_total      += labels.size(0)
        train_correct    += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Tr_Acc: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), train_correct / train_total))
        sys.stdout.flush()
          
def test(epoch):
    global test_loss
    global test_correct
    global test_total
    global best_acc
    net.eval()
    test_loss = 0
    test_correct   = 0
    test_total     = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        
        test_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        test_total      += labels.size(0)
        test_correct    += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Te_Acc: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(test_dataset) // 80, 
                          test_loss / (100 * (idx + 1)), test_correct / test_total))
        sys.stdout.flush()
        
    if test_correct / test_total > best_acc:
        print()
        print('Saving Model...')
        state = {
             'net': net.module if is_use_cuda else net,
             'net_state_dict': net.state_dict(),
#             'acc': test_correct / test_total,
#             'optimizer_state_dict': optimizer.state_dict()
        }
        if not os.path.isdir('./checkpoint/Sqnet_1x_v1.1_elu'):
            os.makedirs('./checkpoint/Sqnet_1x_v1.1_elu')
        torch.save(state, './checkpoint/Sqnet_1x_v1.1_elu/Sqnet_1x_v1.1_elu_Cifar10.ckpt')
        best_acc = test_correct / test_total 
#        checkpoint = torch.load('./checkpoint/Sqnet_1x_v1.0/Sqnet_1x_v1.0_Cifar10.ckpt')
#        net.load_state_dict(checkpoint['net_state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
liveloss = PlotLosses()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    print()
    test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))
    best_cost = end_time - start_time
    if end_time - start_time < best_cost:
        best_cost = end_time - start_time  
              
    liveloss.update({
        'log loss': train_loss,
        'val_log loss': test_loss,
        'accuracy': train_correct,
        'val_accuracy': test_correct
    })
    liveloss.draw()
print('Best Cost: %ds' % (best_cost))    
print('Best Acc: %.4f percent' % (best_acc * 100))


    
    
