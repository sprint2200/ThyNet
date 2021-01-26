import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import models
import torch.nn.functional as F
from resnext_features import resnext101_32x4d_features
from resnext_features import resnext101_64x4d_features



class Resnet101(nn.Module):
  def __init__(self, num_classes =2 ):
    super(Resnet101, self).__init__()
    model_resnet101 = models.resnet101(pretrained=True)
    self.conv1 = model_resnet101.conv1
    self.bn1 = model_resnet101.bn1
    self.relu = model_resnet101.relu
    self.maxpool = model_resnet101.maxpool
    self.layer1 = model_resnet101.layer1
    self.layer2 = model_resnet101.layer2
    self.layer3 = model_resnet101.layer3
    self.layer4 = model_resnet101.layer4
    self.avgpool = model_resnet101.avgpool
    self.__in_features = model_resnet101.fc.in_features
    self.fc = nn.Linear(2048, num_classes)
    # self.softmax = nn.Softmax()
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
    x = self.fc(x)
    # x = self.softmax(x)
    return x

  def output_num(self):
    return self.__in_features

class Densnet201(nn.Module):
  def __init__(self, num_classes =2):
    super(Densnet201, self).__init__()
    model_densenet201 = models.densenet201(pretrained=True)
    self.features = model_densenet201.features
    self.fc = nn.Linear(1920, num_classes)
    self.relu_out = 0

  def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    self.relu_out = out
    out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
    out = self.fc(out)
    return out

  def cam_out(self):
      return self.relu_out

class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

class Resnext101(nn.Module):
    def __init__(self, num_classes =2):
        super(Resnext101, self).__init__()
        use_model = ResNeXt101_32x4d()
        use_model.load_state_dict(torch.load('./model/resnext101_32x4d-29e315fa.pth'))
        self.features = use_model.features
        self.avg_pool = use_model.avg_pool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x










