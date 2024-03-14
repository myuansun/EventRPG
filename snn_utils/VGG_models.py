import math
import torch
import torch.nn as nn
from snn_utils.tet_layers import *


class VGGSNN(nn.Module):
    def __init__(self, resolution=(48, 48), num_classes=10, T=10, relprop_mode='sltrp'):
        super(VGGSNN, self).__init__()
        self.num_classes = num_classes
        self.relprop_mode = relprop_mode
        self.features = Sequential(
            Layer(2,64,3,1,1, T=T, relprop_mode=relprop_mode),
            Layer(64,128,3,1,1, T=T, relprop_mode=relprop_mode),
            SeqToANNContainer(AvgPool2d(2, T=T, relprop_mode=relprop_mode)),
            Layer(128,256,3,1,1, T=T, relprop_mode=relprop_mode),
            Layer(256,256,3,1,1, T=T, relprop_mode=relprop_mode),
            SeqToANNContainer(AvgPool2d(2, T=T, relprop_mode=relprop_mode)),
            Layer(256,512,3,1,1, T=T, relprop_mode=relprop_mode),
            Layer(512,512,3,1,1, T=T, relprop_mode=relprop_mode),
            SeqToANNContainer(AvgPool2d(2, T=T, relprop_mode=relprop_mode)),
            Layer(512,512,3,1,1, T=T, relprop_mode=relprop_mode),
            Layer(512,512,3,1,1, T=T, relprop_mode=relprop_mode)
        )
        self.avg_pool = SeqToANNContainer(AvgPool2d(2, T=T, relprop_mode=relprop_mode))
        W = int(resolution[0]/2/2/2/2)
        self.classifier = SeqToANNContainer(Linear(512*W*W, num_classes, T=T, relprop_mode=relprop_mode))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input, str_target_layer="forward", alpha=1, labels=None):
        feature_map = self.features(input)
        pool_feature = self.avg_pool(feature_map)
        feature_shape = pool_feature.shape
        flatten_features = torch.flatten(pool_feature, 2)
        x = self.classifier(flatten_features)
        if str_target_layer == "forward":
            return x
        R = self.CLRP(x, labels)
        R = self.classifier.relprop(R, alpha)
        if self.relprop_mode == "slrp":
            feature_shape = [*feature_shape]
            feature_shape[1] = 1
        R = R.view(feature_shape)
        R = self.avg_pool.relprop(R, alpha)
        if str_target_layer == "SAM_layer4":
            R = torch.zeros_like(feature_map)
            for T in range(feature_map.shape[1]):
                for t in range(T):
                    R[:, T] += feature_map[:, t] * math.exp(-0.2 * (T - t))
            R = R.sum(dim=2, keepdim=True)
            return R, x
        if str_target_layer == "RelCAM_layer4":
            if self.relprop_mode == "sltrp":
                r_weight = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam = feature_map * r_weight
                r_cam = torch.sum(r_cam, dim=(2), keepdim=True)
                return r_cam, x
            elif self.relprop_mode == "slrp":
                r_weight = torch.sum(R, dim=(3, 4), keepdim=True)
                r_cam = feature_map * r_weight
                r_cam = torch.sum(r_cam, dim=(2), keepdim=True)
                return r_cam, x
        if str_target_layer == "layer4":
            if self.relprop_mode == "sltrp":
                return R.mean(2, keepdim=True), x
            elif self.relprop_mode == "slrp":
                return R.mean(2, keepdim=True), x
        elif str_target_layer == "long":
            R = self.features.relprop(R, alpha)
            return R, x


    def CLRP(self, x, max_index=None):
        if max_index is None:
            max_index = torch.argmax(x.mean(1), dim=1)
        R = torch.ones(x.shape).to(x.device)
        R /= -self.num_classes
        for j in range(R.size(0)):
            R[j, :, max_index[j]] += 1
        if self.relprop_mode == "sltrp":
            return R
        elif self.relprop_mode == "slrp":
            return R.mean(1, keepdim=True)

    