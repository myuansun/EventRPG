import torch
import torch.nn as nn
from copy import deepcopy
from snn_utils.layers import *
import math

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
           'sew_resnet152', 'sew_resnext50_32x4d', 'sew_resnext101_32x8d',
           'sew_wide_resnet50_2', 'sew_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def sew_function(cnf:str, T=16, relprop_mode='sltrp'):
    if cnf == 'ADD':
        return Add(T=T, relprop_mode=relprop_mode)
    elif cnf == 'AND':
        return Multiply()
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, T=16, relprop_mode="sltrp"):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, groups=groups, bias=False, dilation=dilation, T=T, relprop_mode=relprop_mode)


def conv1x1(in_planes, out_planes, stride=1, T=16, relprop_mode="sltrp"):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, T=T, relprop_mode=relprop_mode)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, spiking_neuron: callable = None,
                 **kwargs):
        super(BasicBlock, self).__init__()
        T = kwargs['T']
        relprop_mode = kwargs['relprop_mode']
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.cnf = cnf
        self.sew_function = sew_function(self.cnf, T=T, relprop_mode=relprop_mode)
        self.clone = Clone(T=T, relprop_mode=relprop_mode)
        self.conv1 = conv3x3(inplanes, planes, stride, T=T, relprop_mode=relprop_mode)
        self.bn1 = norm_layer(planes, T=T, relprop_mode=relprop_mode)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes, T=T, relprop_mode=relprop_mode)
        self.bn2 = norm_layer(planes, T=T, relprop_mode=relprop_mode)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = spiking_neuron(**deepcopy(kwargs))
        self.stride = stride


    def forward(self, x):
        x, identity = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = self.sew_function([out, identity])

        return out

    def relprop(self, R, alpha):
        out, x2 = self.sew_function.relprop(R, alpha)
        if self.downsample is not None:
            x2 = self.downsample_sn.relprop(x2, alpha)
            x2 = self.downsample.relprop(x2, alpha)

        out = self.sn2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.sn1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, spiking_neuron: callable = None,
                 **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.cnf = cnf
        self.sew_function = sew_function(self.cnf)
        self.clone = Clone()
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = spiking_neuron(**deepcopy(kwargs))
        self.stride = stride

    def forward(self, x):
        x, identity = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = self.sew_function([out, identity])

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str = None, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        T = kwargs['T']
        relprop_mode=kwargs['relprop_mode']
        self.relprop_mode = relprop_mode
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, T=T, relprop_mode=relprop_mode)
        self.bn1 = norm_layer(self.inplanes, T=T, relprop_mode=relprop_mode)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = MaxPool2d(3, 2, 1, T=T, relprop_mode=relprop_mode)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.avgpool = AdaptiveAvgPool2d((1, 1), T=T, relprop_mode=relprop_mode)
        self.fc = Linear(512 * block.expansion, num_classes, T=T, relprop_mode=relprop_mode)
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None,
                    spiking_neuron: callable = None, **kwargs):
        T = kwargs['T']
        relprop_mode = kwargs['relprop_mode']
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, T=T, relprop_mode=relprop_mode),
                norm_layer(planes * block.expansion, T=T, relprop_mode=relprop_mode),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))

        return Sequential(*layers)

    def _forward_impl(self, x, str_target_layer, labels=None, alpha=1):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        if self.relprop_mode == 'sltrp':
            avg_shape = x.shape
        elif self.relprop_mode == 'slrp':
            avg_shape = [1] + list(x.shape[1:])
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)

        x = self.fc(x)

        if str_target_layer == "forward":
            return x
        elif "layer4" in str_target_layer:
            return self.relprop(x, alpha, avg_shape, layer4, str_target_layer, labels=labels)
        elif "layer3" in str_target_layer:
            return self.relprop(x, alpha, avg_shape, layer3, str_target_layer, labels=labels)
        elif "layer2" in str_target_layer:
            return self.relprop(x, alpha, avg_shape, layer2, str_target_layer, labels=labels)
        elif "layer1" in str_target_layer:
            return self.relprop(x, alpha, avg_shape, layer1, str_target_layer, labels=labels)
        elif str_target_layer == "long":
            return self.relprop(x, alpha, avg_shape, str_target_layer=str_target_layer, labels=labels)
        else:
            raise Exception
        return x

    def forward(self, x, str_target_layer="forward", labels=None, alpha=1):
        return self._forward_impl(x, str_target_layer, labels, alpha)

    def CLRP(self, x, max_index=None):
        # x = x.sum(0)
        if max_index is None:
            max_index = torch.argmax(x.mean(0), dim=1)
        R = torch.ones(x.shape).to(x.device)
        R /= -self.num_classes
        for j in range(R.size(0)):
            for i in range(R.size(1)):
                R[j, i, max_index[i]] += 1
        if self.relprop_mode == "sltrp":
            return R
        elif self.relprop_mode == "slrp":
            return R.mean(0, keepdim=True)

    def relprop(self, y, alpha, avg_shape, feature_map=None, str_target_layer="long", labels=None):
        if "SAM" in str_target_layer:
            R = torch.zeros_like(feature_map)
            for T in range(feature_map.shape[0]):
                for t in range(T):
                    R[T] += feature_map[t] * math.exp(-0.2 * (T - t))
            R = R.sum(dim=2, keepdim=True)#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
            return R, y
        R = self.CLRP(y, labels)
        R = self.fc.relprop(R, alpha)
        R = R.view(avg_shape)
        R = self.avgpool.relprop(R, alpha)
        if "layer4" in str_target_layer:
            if "RelCAM" in str_target_layer:
                r_weight4 = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam4 = feature_map * r_weight4
                r_cam4 = torch.sum(r_cam4, dim=(2), keepdim=True)
                r_cam4 = r_cam4#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
                return r_cam4, y
            else:
                return R.mean(2, keepdim=True), y
            
        R = self.layer4.relprop(R, alpha)
        if "layer3" in str_target_layer:
            if "RelCAM" in str_target_layer:
                r_weight3 = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam3 = feature_map * r_weight3
                r_cam3 = torch.sum(r_cam3, dim=(2), keepdim=True)
                r_cam3 = r_cam3#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
                return r_cam3, y
            else:
                return R.mean(2, keepdim=True), y
        R = self.layer3.relprop(R, alpha)
        if "layer2" in str_target_layer:
            if "RelCAM" in str_target_layer:
                r_weight2 = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam2 = feature_map * r_weight2
                r_cam2 = torch.sum(r_cam2, dim=(2), keepdim=True)
                r_cam2 = r_cam2#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
                return r_cam2, y
            else:
                return R.mean(2, keepdim=True), y
        R = self.layer2.relprop(R, alpha)
        if "layer1" in str_target_layer:
            if "RelCAM" in str_target_layer:
                r_weight1 = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam1 = feature_map * r_weight1
                r_cam1 = torch.sum(r_cam1, dim=(2), keepdim=True)
                r_cam1 = r_cam1#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
                return r_cam1, y
            else:
                return R.mean(2, keepdim=True), y
        elif str_target_layer == "long":
            R = self.layer1.relprop(R, alpha)
            R = self.maxpool.relprop(R, alpha)
            R = self.sn1.relprop(R, alpha)
            R = self.bn1.relprop(R, alpha)
            R = self.conv1.relprop(R, alpha)
            R = R#.permute(1, 0, 2, 3, 4).contiguous() #B, T, C, H, W
            return R, y
        else:
            raise Exception


def _sew_resnet(arch, block, layers, pretrained, progress, cnf, spiking_neuron, **kwargs):
    model = SEWResNet(block, layers, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def sew_resnet18(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-18 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _sew_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet34(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-34 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _sew_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet50(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-50 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _sew_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet101(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-101 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _sew_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet152(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-152 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _sew_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnext50_32x4d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    The spike-element-wise ResNeXt-50 32x4d `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNeXt-50 32x4d model from `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _sew_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnext101_32x8d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    The spike-element-wise ResNeXt-101 32x8d `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNeXt-101 32x8d model from `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _sew_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron,
                       **kwargs)


def sew_wide_resnet50_2(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    The spike-element-wise Wide ResNet-50-2 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the Wide ResNet-50-2 model from `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_wide_resnet101_2(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    The spike-element-wise Wide ResNet-101-2 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the Wide ResNet-101-2 model from `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron,
                       **kwargs)
