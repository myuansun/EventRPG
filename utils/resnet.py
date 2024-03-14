import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from utils.layers import *
import torch
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.clone = Clone()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.relu2(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu2.relprop(R, alpha)
        out, x2 = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x2 = self.downsample.relprop(x2, alpha)

        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.clone = Clone()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.relu3 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        # out = self.add([out, x2])
        out = self.add([out, x])
        out = self.relu3(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu3.relprop(R, alpha)

        out, x = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.relu2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, long= False, zero_init_residual=False, device=torch.device('cuda')):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)
        self.long = long
        self.num_classes = num_classes
        self.device = device

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def CLRP(self, x, maxindex=None):
        if maxindex == None:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).to(self.device)
        R /= - (self.num_classes - 1)
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R

    def time_cam_forward(self, x):
        R_cam = []
        for i in range(x.shape[1]):
            single_x = torch.zeros_like(x)
            single_x[0, :] = x[0, i].unsqueeze(0).repeat(x.shape[1], 1, 1)
            r_cam, z = self.forward(single_x, 'layer4')
            R_cam.append(r_cam)
        R_cam = torch.cat(R_cam, dim=1)
        return R_cam, z

    def forward(self, x, mode='output', labels=None, alpha=2):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        avgpool_shape = x.shape
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        if mode == 'output':
            return z

        R = self.CLRP(z, labels)
        if mode == 'long':
            self.long = True
            layer_weight, raw_R = self.relprop(R, 1, avgpool_shape)
            return raw_R, z
        R = self.fc.relprop(R, alpha)
        R = R.reshape(*avgpool_shape)
        R4 = self.avgpool.relprop(R, alpha)
        if mode == 'layer4':
            r_weight4 = torch.mean(R4, dim=(2, 3), keepdim=True)
            r_cam4 = layer4 * r_weight4
            r_cam4 = torch.sum(r_cam4, dim=(1), keepdim=True)
            return r_cam4, z

        elif mode == 'layer3':
            R3 = self.layer4.relprop(R4, alpha)
            r_weight3 = torch.mean(R3, dim=(2, 3), keepdim=True)
            r_cam3 = layer3 * r_weight3
            r_cam3 = torch.sum(r_cam3, dim=(1), keepdim=True)
            return r_cam3, z
        elif mode == 'layer2':
            R3 = self.layer4.relprop(R4, alpha)
            R2 = self.layer3.relprop(R3, alpha)
            r_weight2 = torch.mean(R2, dim=(2, 3), keepdim=True)
            r_cam2 = layer2 * r_weight2
            r_cam2 = torch.sum(r_cam2, dim=(1), keepdim=True)
            return r_cam2, z
        elif mode == 'layer1':
            R3 = self.layer4.relprop(R4, alpha)
            R2 = self.layer3.relprop(R3, alpha)
            R1 = self.layer2.relprop(R2, alpha)
            r_weight1 = torch.mean(R1, dim=(2, 3), keepdim=True)
            r_cam1 = layer1 * r_weight1
            r_cam1 = torch.sum(r_cam1, dim=(1), keepdim=True)
            return r_cam1, z
        else:
            return z

    def relprop(self, R, alpha, avgpool_shape):
        R = self.fc.relprop(R, alpha)
        R = R.reshape(avgpool_shape)
        R = self.avgpool.relprop(R, alpha)
        layer_weight = torch.sum(R, dim=(2, 3), keepdim=True)
        R = self.layer4.relprop(R, alpha)
        R = self.layer3.relprop(R, alpha)
        R = self.layer2.relprop(R, alpha)
        R = self.layer1.relprop(R, alpha)
        R = self.maxpool.relprop(R, alpha)
        R = self.relu.relprop(R, alpha)
        R = self.bn1.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)
        return layer_weight, R


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, device=torch.device("cuda"), **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param device: current device
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, long = False,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], long = long,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
