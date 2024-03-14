import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import numpy as np
from tqdm import tqdm
from snn_utils.layers import *
from snn_utils import sew_resnet
from spikingjelly.activation_based import functional
from snn_utils.VGG_models import VGGSNN
from torchvision import transforms

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()
            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


class QuantizationLayerEST(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1 + events[-1, -1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()


        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t - i_bin / (C - 1))

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim, mode):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = mode

    def forward(self, events):
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T
        x = x.to(torch.int64)
        y = y.to(torch.int64)
        p = p.to(torch.int64)
        b = b.to(torch.int64)
        # normalizing timestamps
        unit_len = []
        t_idx = []
        for bi in range(B):
            bi_idx = events[:, -1] == bi
            t[bi_idx] /= t[bi_idx].max()
            unit_len.append(int(bi_idx.float().sum() / C))
            _, t_idx_ = torch.sort(t[events[:, -1] == bi])
            t_idx.append(t_idx_)
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b


        for i_bin in range(C):
            values = torch.zeros_like(t)
            for bi in range(B):
                bin_idx = t_idx[bi][i_bin * unit_len[bi]: (i_bin + 1) * unit_len[bi]]
                bin_values = values[events[:, -1] == bi]
                bin_values[bin_idx] = 1
                values[events[:, -1] == bi] = bin_values
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)#.clamp(0, 1)
        if self.mode == "TB":
            vox = vox.permute(2, 0, 1, 3, 4).contiguous()
        elif self.mode == "BT":
            vox = vox.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise Exception
        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9, 180, 240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 pretrained=True,
                 device=torch.device("cuda"),
                 classifier="resnet34",
                 spiking_neuron="PLIF",
                 T=16,
                 relprop_mode="sltrp"):

        nn.Module.__init__(self)
        if spiking_neuron == "IF":
            neuron = IFNode
        elif spiking_neuron == "LIF":
            neuron = LIFNode
        elif spiking_neuron == "PLIF":
            neuron = ParametricLIFNode
        else:
            raise Exception
        if classifier == "resnet34":
            self.classifier = sew_resnet.sew_resnet34(pretrained=pretrained, progress=True, spiking_neuron=neuron, cnf="ADD", num_classes=1000, T=voxel_dimension[0], relprop_mode=relprop_mode)
            mode = 'TB'
        elif classifier == "resnet18":
            self.classifier = sew_resnet.sew_resnet18(pretrained=pretrained, progress=True, spiking_neuron=neuron, cnf="ADD", num_classes=1000, T=voxel_dimension[0], relprop_mode=relprop_mode)
            mode = 'TB'
        elif classifier == "vgg11":
            self.classifier = VGGSNN(crop_dimension, num_classes, relprop_mode=relprop_mode)
            mode = 'BT'
        self.quantization_layer = QuantizationLayerVoxGrid(voxel_dimension, mode)
        self.device = device
        self.crop_dimension = crop_dimension
        self.transform = transforms.Resize(self.crop_dimension)
        # replace fc layer and first convolutional layer
        if "resnet" in classifier:
            self.classifier.conv1 = Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False, T=voxel_dimension[0], relprop_mode=relprop_mode)
            self.classifier.fc = Linear(self.classifier.fc.in_features, num_classes, T=voxel_dimension[0], relprop_mode=relprop_mode)
            functional.set_step_mode(self.classifier, "m")
            self.classifier.num_classes = num_classes



    def resize_to_resolution(self, x):
        B, T, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y_shape = [x.shape[i] for i in range(3)]
        y = ZeroPad(x)
        y_shape.extend(list(self.crop_dimension))
        y = self.transform(y.flatten(0, 1)).view(y_shape)
        # y = F.interpolate(y, size=(2, *self.crop_dimension))
        return y

    def pad_to_resolution(self, x):
        B, T, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y_shape = [x.shape[i] for i in range(3)]
        y = ZeroPad(x)
        return y

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_resized = self.resize_to_resolution(vox)
        pred = self.classifier.forward(vox_resized)
        functional.reset_net(self.classifier)
        return pred




