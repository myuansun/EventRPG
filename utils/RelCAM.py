import numpy as np
import torch
import cv2
from utils.layers import safe_divide
from spikingjelly.activation_based import functional

def normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA = safe_divide(AA, AA.max(1, keepdim=True)[0])
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac

def percentile(t, q):
    B, C, T, H, W = t.shape
    k = 1 + round(.01 * float(q) * (H * W - 1))
    result = t.reshape(B, C, T, -1).kthvalue(k).values
    return result[:, :, :, None, None]

def mid_normalize(representation):
    if len(representation.shape) == 4:
        representation = representation.unsqueeze(1)
    B, T, C, H, W = representation.shape
    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99) + 1e-9
    robust_mid_vals = torch.mode(representation.view(B, T, C, -1), dim=3)[0].view(B, T, C, 1, 1)
    robust_min_vals = percentile(representation, 1) - 1e-9
    representation = (representation - robust_mid_vals) / (
                (robust_max_vals - robust_mid_vals) * (representation > robust_mid_vals) + (
                    robust_mid_vals - robust_min_vals) * (representation <= robust_mid_vals))
    representation = representation.clamp(min=-1, max=1) * 0.5 + 0.5
    return representation

def tensor2image(x, i=0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = cv2.resize(np.transpose(x, (1, 2, 0)), (224, 224))
    return x


def threshold(x, std_ratio=0):
    # print(f"nan number {torch.isnan(x).sum()}")
    mean_ = x.mean()
    std_ = x.std()
    thresh = mean_ + std_ * std_ratio
    if thresh > x.max():
        thresh = x.max()
    x = (x >= thresh)
    return x


class HookRecord:
    def __init__(self, target_layer):
        self.value = dict()

        def forward_hook(module, input, output):
            self.value['activations'] = output

        def backward_hook(module, input, output):
            self.value['gradients'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __getitem__(self, key):
        return self.value[key]


class RelavanceCAM:
    def __init__(self, model, resolution, mask_std_ratio, coord_std_ratio, alpha, data_shape_prefix, device=torch.device("cuda")):
        self.model = model
        self.score_cam_classes = []
        self.device = device
        self.resolution = resolution
        self.mask_std_ratio = mask_std_ratio
        self.coord_std_ratio = coord_std_ratio
        self.alpha = alpha
        self.data_shape_prefix = data_shape_prefix

    def get_cam_image(self, events, str_target_layer='long', labels=None):
        self.model.eval()
        events = events.to(self.model.device)
        representation = self.model.quantization_layer.forward(events)
        vox_resized = self.model.resize_to_resolution(representation)
        R_CAM, output = self.model.classifier(vox_resized, str_target_layer, labels=labels, alpha=self.alpha)
        if len(R_CAM.shape) == 5:
            functional.reset_net(self.model.classifier)
            ## gesture model
        if self.data_shape_prefix == "TB":
            R_CAM = R_CAM.permute(1, 0, 2, 3, 4).contiguous() # B, T, C, H, W
        return R_CAM

    def get_threshold(self, events, str_target_layer='layer4'):
        H, W = self.resolution
        R_CAM = self.get_cam_image(events, str_target_layer)
        if len(R_CAM.shape) == 5:
            R_CAM = R_CAM.sum(1)  # B,1,H,W
        R_CAM = normalize(R_CAM)

        R_CAM = R_CAM.detach()
        R_CAM = torch.nn.functional.interpolate(R_CAM, (max(H, W), max(H, W)), mode="bilinear")
        cam_shape = R_CAM.shape
        if H < W:
            a = (1 - H / W) / 2
            R_CAM = R_CAM[:, :, round(cam_shape[2] * a):round(cam_shape[2] * (1 - a)), :]
        elif H > W:
            a = (1 - W / H) / 2
            R_CAM = R_CAM[:, :, :, round(cam_shape[3] * a):round(cam_shape[3] * (1 - a))]
        coordinates = []
        for b in range(R_CAM.shape[0]):
            single_R_CAM = R_CAM[b].view(self.resolution)
            if torch.isnan(single_R_CAM).sum() != 0:
                coordinates.append([0, 0, W - 1, H - 1])
            else:
                thre = threshold(single_R_CAM).cpu()
                thre = np.where(thre == 1)
                x_min, y_min = thre[1].min(), thre[0].min()
                x_max, y_max = thre[1].max(), thre[0].max()
                coordinates.append([x_min, y_min, x_max, y_max])
        return coordinates

    def get_heat_prob(self, events, str_target_layer='layer4'):
        H, W = self.resolution
        R_CAM = self.get_cam_image(events, str_target_layer)
        R_CAM = normalize(R_CAM)
        R_CAM = R_CAM.detach()
        if len(R_CAM.shape) == 5:
            R_CAM = R_CAM.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
            if R_CAM.shape[1] != 1:
                R_CAM = R_CAM.flatten(1, 2)
            else:
                R_CAM = R_CAM.mean(2)
        if not R_CAM.shape[3] == self.resolution[1]:
            R_CAM = torch.nn.functional.interpolate(R_CAM, (max(H, W), max(H, W)), mode="bilinear")
        R_CAM = R_CAM.squeeze(1)
        return R_CAM.to(events.device)

    def get_mask(self, events, str_target_layer='layer4', labels=None):
        H, W = self.resolution
        R_CAM = self.get_cam_image(events, str_target_layer, labels).detach()
        if str_target_layer == 'long':
            R_CAM = mid_normalize(R_CAM)
        elif str_target_layer == 'layer4':
            R_CAM = normalize(R_CAM)
        if len(R_CAM.shape) == 5:
            R_CAM = R_CAM.mean((1, 2)).unsqueeze(1) # (B,1,H,W)
        R_CAM = torch.nn.functional.interpolate(R_CAM, (max(H, W), max(H, W)), mode="bilinear")
        cam_shape = R_CAM.shape
        if H < W:
            a = (1 - H / W) / 2
            R_CAM = R_CAM[:, :, round(cam_shape[2] * a):round(cam_shape[2] * (1 - a)), :]
        elif H > W:
            a = (1 - W / H) / 2
            R_CAM = R_CAM[:, :, :, round(cam_shape[3] * a):round(cam_shape[3] * (1 - a))]
        mask_thre = []
        coordinates = []
        for b in range(R_CAM.shape[0]):
            single_R_CAM = R_CAM[b].view(self.resolution)
            if torch.isnan(single_R_CAM).sum() != 0:
                mask_thre.append(torch.zeros_like(single_R_CAM).unsqueeze(0))
                coordinates.append(torch.tensor([0, 0, W - 1, H - 1]).unsqueeze(0))
            else:
                thre_ = threshold(single_R_CAM, self.mask_std_ratio)
                thre = threshold(single_R_CAM, self.coord_std_ratio)
                thre = np.where(thre.cpu().numpy() == 1)
                x_min, y_min = thre[1].min(), thre[0].min()
                x_max, y_max = thre[1].max(), thre[0].max()
                mask_thre.append(thre_.unsqueeze(0))
                coordinates.append(torch.tensor([x_min, y_min, x_max, y_max]).unsqueeze(0))
        mask_thre = torch.cat(mask_thre, dim=0)
        coordinates = torch.cat(coordinates, dim=0)
        return mask_thre, coordinates
