import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name, pretrain=True):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=pretrain)
        # model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    # args里默认为wn
    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x

class feat_classifier_simpl(nn.Module):
    def __init__(self, class_num, feat_dim):
        super(feat_classifier_simpl, self).__init__()
        self.fc = nn.Linear(feat_dim, class_num)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

# 从噪声投回去的使特征一致性的网络，输入2048,输出2048或256。暂定2048
# 扩散模型网络？？
# VAE网络？？
# 线性网络？？
class feat_consistency(nn.Module):
    def __init__(self, feature_dim, channel_out= 2048, type = 'Linear'):
        super(feat_consistency, self).__init__()
        self.feature_dim = feature_dim
        self.channel_out = channel_out
        if type == 'Resnet':
            self.F1 = subnet_constructor('Resnet', None, self.feature_dim,channel_out)
            self.F2 = subnet_constructor('Resnet', None, self.feature_dim,channel_out)
            self.F3 = subnet_constructor('Resnet', None, self.feature_dim,channel_out)
        elif type == 'Linear':
            self.F1 = nn.Linear(feature_dim, channel_out)
            self.F2 = nn.ReLU(inplace= False)
            self.F3 = nn.Linear(feature_dim, channel_out)
        else:
            pass

    def forward(self, x):
       # x = x.unsqueeze(2).unsqueeze(3)
        x = self.F1(x)
        x = self.F2(x)
        x = self.F3(x)
        #x = x.squeeze()
        return x
        


def subnet_constructor(net_structure, init, channel_in, channel_out):
    if net_structure == 'DBNet':
        if init != 'xavier':
            return DenseBlock(channel_in, channel_out, init)
        else:
            return DenseBlock(channel_in, channel_out)
    elif net_structure == 'Resnet':
        return ResBlock(channel_in, channel_out)
    elif net_structure == 'Linear':
        return
    else:
        return None


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out

