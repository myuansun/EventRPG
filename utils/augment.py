import copy
import time
import numpy as np
import math
import random
import torch
from utils.RelCAM import RelavanceCAM
import torchvision


def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:, None, None, None]


def create_image(representation):
    B, C, H, W = representation.shape
    representation = representation.view(B, 3, C // 3, H, W).sum(2)
    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)
    representation = (representation - robust_min_vals) / (robust_max_vals - robust_min_vals)
    representation = torch.clamp(255 * representation, 0, 255).byte()
    representation = torchvision.utils.make_grid(representation)
    img = np.array(representation.permute(1, 2, 0))
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_show


def visualize_events(events, model):
    events = torch.cat([events, 0 * torch.ones((len(events), 1), dtype=torch.float32).to(events.device)], 1)
    representation = model.quantization_layer(events)
    img_show = create_image(representation)
    # cv2.imwrite(path, img_show)
    return img_show


class EventAugment(object):
    def __init__(self, resolution, model, l_mags=11, mask_std_ratio=0.5, coord_std_ratio=1, relevance_mix='layer4', alpha=1, data_shape_prefix=None, device=torch.device("cuda")):
        self.resolution = resolution
        self.rel_cam = RelavanceCAM(model, resolution, mask_std_ratio, coord_std_ratio, alpha, data_shape_prefix, device=device)
        self.relevance_mix = relevance_mix
        self.data_shape_prefix = data_shape_prefix
        self.augment_list = [
                             (self.identity, 0, 0),
                             (self.drop_by_area, 0.1, 0.5),
                             (self.random_drop_with_cam, 0.5, 1),
                             (self.flip_along_x, 0, 0),
                             (self.rotate, -math.pi/3, math.pi/3),
                             (self.linear_x, -0.3, 0.3),
                             (self.linear_y, -0.3, 0.3),
                             (self.shear_x, -0.3, 0.3),
                             (self.shear_y, -0.3, 0.3),
                             (self.scale, 0.5, 1.5)
        ]
        self.device = device
        self.ops_name = []
        self.l_ops = len(self.augment_list)
        self.l_mags = l_mags
        for idx, op in enumerate(self.augment_list):
            self.ops_name.append(op.__str__().split(' ')[2].split('.')[1])


    def batch_augment(self, events, labels, mask, mix_prob=0.5, op_idx=None):


        aug_events = []
        B = int(1 + events[-1, -1].item())
        layer4_cam_probs = None
        if op_idx is None:
            op_index = random.randint(0, len(self.augment_list) - 1)
        else:
            op_index = op_idx
        if "cam" in self.ops_name[op_index]:
            long_cam_probs = self.rel_cam.get_heat_prob(events, str_target_layer='long')
            aug_op, min_mag, max_mag = self.augment_list[op_index]
            for b in range(B):
                single_events = events[events[:, 4] == b, :4]
                mag = random.random()
                aug_mag = (max_mag - min_mag) * mag + min_mag
                aug_single_events = aug_op(single_events, aug_mag, layer4_cam_probs, long_cam_probs[b])
                aug_single_events = torch.cat([aug_single_events, b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
                aug_events.append(aug_single_events)
        else:
            for b in range(B):
                single_events = events[events[:, 4] == b, :4]
                if op_idx is None:
                    op_index = random.randint(0, len(self.augment_list) - 1)
                    while "cam" in self.ops_name[op_index]:
                        op_index = random.randint(0, len(self.augment_list) - 1)
                else:
                    op_index = op_idx
                mag = random.random()
                aug_op, min_mag, max_mag = self.augment_list[op_index]
                aug_mag = (max_mag - min_mag) * mag + min_mag
                aug_single_events = aug_op(single_events, aug_mag)
                assert aug_single_events.shape[0] > 0, "shape {} with ops {} mags {}".format(aug_single_events.shape, self.ops_name[op_index],
                                                                                    aug_mag)
                aug_single_events = torch.cat([aug_single_events, b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
                aug_events.append(aug_single_events)
        aug_events = torch.cat(aug_events, dim=0)
        if random.random() < mix_prob:
            return aug_events, labels, False
        else:
            if mask == "mask":
                aug_events, labels = self.mixup_with_cam(events, labels, True, True)
            elif mask == "pre":
                aug_events, labels = self.mixup_with_cam(events, labels, True, False)
            elif mask == "post":
                aug_events, labels = self.mixup_with_cam(events, labels, False, True)
            elif mask == "no":
                aug_events, labels = self.mixup_with_cam(events, labels, False, False)
            else:
                raise Exception
            return aug_events, labels, True


    def nda_batch_augment(self, events):
        # aug_events = events
        # NDA Aug

        aug_events = []
        B = int(1 + events[-1, -1].item())
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            aug_single_events = self.nda(single_events)
            aug_single_events = torch.cat(
                [aug_single_events,
                 b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
            aug_events.append(aug_single_events)
        aug_events = torch.cat(aug_events, dim=0)
        return aug_events

    def eventdrop_batch_augment(self, events, option=None):
        # aug_events = events
        # NDA Aug

        aug_events = []
        B = int(1 + events[-1, -1].item())
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            aug_single_events = self.event_drop(single_events, option)
            aug_single_events = torch.cat(
                [aug_single_events,
                 b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
            aug_events.append(aug_single_events)
        aug_events = torch.cat(aug_events, dim=0)
        return aug_events

    def identity(self, events, v, layer4=None, long=None):
        events = copy.deepcopy(events)
        return events

    def drop_by_time(self, events, ratio):
        events = copy.deepcopy(events)
        timestamps = events[:, 2]
        t_max = timestamps.max()
        t_min = timestamps.min()
        t_period = t_max - t_min
        drop_period = t_period * ratio
        t_start = torch.rand(1).to(self.device) * (t_max - drop_period - t_min) + t_min
        t_end = t_start + drop_period
        idx = (timestamps < t_start) | (timestamps > t_end)
        return events[idx]

    def drop_by_area(self, events, area_ratio):
        events = copy.deepcopy(events)
        length_scale = torch.rand(1).to(self.device) + 0.5
        x0 = np.random.uniform(self.resolution[1])
        y0 = np.random.uniform(self.resolution[0])
        x_out = self.resolution[1] * area_ratio * length_scale
        y_out = self.resolution[0] * area_ratio / length_scale
        x0 = int(max(0, x0 - x_out / 2.0))
        y0 = int(max(0, y0 - y_out / 2.0))
        x1 = min(self.resolution[1], x0 + x_out)
        y1 = min(self.resolution[0], y0 + y_out)
        xy = (x0, x1, y0, y1)
        idx1 = (events[:, 0] < xy[0]) | (events[:, 0] > xy[1])
        idx2 = (events[:, 1] < xy[2]) | (events[:, 1] > xy[3])
        idx = idx1 | idx2
        if events[idx].shape[0] != 0:
            return events[idx]
        else:
            return events

    def random_drop(self, events, ratio):
        if ratio == 1:
            ratio -= 0.1
        events = copy.deepcopy(events)
        N = events.shape[0]
        num_drop = int(N * ratio)
        idx = random.sample(list(np.arange(0, N)), N - num_drop)
        return events[idx]

    def random_drop_with_cam(self, events, lamda, layer4_cam_prob, long_cam_prob):
        single_events = copy.deepcopy(events)
        cam_prob = long_cam_prob * lamda
        rand = torch.rand(len(single_events)).to(single_events.device)
        int_events = single_events[:, :2].to(torch.int64)
        t_max, t_min = single_events[:, 2].max(), single_events[:, 2].min()
        num_channel = int(cam_prob.shape[0] / 2)
        len_channel = ((t_max - t_min) / num_channel).item()
        t = ((single_events[:, 2] - t_min).div(len_channel, rounding_mode='floor')).clamp(max=num_channel - 1).to(
            torch.int64)
        p = single_events[:, 3].to(torch.int64)
        index = rand[:] > cam_prob[t + p * num_channel, int_events[:, 1], int_events[:, 0]]
        if single_events[index].shape[0] != 0:
            single_events = single_events[index]
        return single_events

    def flip_along_x(self, events, v):
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 0] = W - 1 - events[:, 0]
        return events

    def rotate(self, events, theta):
        events = copy.deepcopy(events)
        H, W = self.resolution
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        x = events[:, 0] - x_mid
        y = events[:, 1] - y_mid
        events[:, 0] = torch.round(x * math.cos(theta) + y * math.sin(theta) + x_mid)
        events[:, 1] = torch.round(-x * math.sin(theta) + y * math.cos(theta) + y_mid)
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def linear_x(self, events, linear):
        events = copy.deepcopy(events)
        W = self.resolution[1]
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        x_mid = (x_max + x_min) / 2
        if linear > 0:
            linear_w = int(linear * (W - x_mid))
        else:
            linear_w = int(linear * x_mid)
        events[:, 0] = events[:, 0] + linear_w
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W)
        return events[valid_events]

    def linear_y(self, events, linear):
        events = copy.deepcopy(events)
        H = self.resolution[0]
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        y_mid = (y_max + y_min) / 2
        if linear > 0:
            linear_h = int(linear * (H - y_mid))
        else:
            linear_h = int(linear * y_mid)
        events[:, 1] = events[:, 1] + linear_h
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def shear_y(self, events, shear):
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 1] = torch.round(events[:, 1] + shear * (events[:, 0] - x_mid) / (x_max - x_min) * (y_max - y_min))
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def shear_x(self, events, shear):
        if random.random() < 0.5:
            shear = -shear
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 0] = torch.round(events[:, 0] + shear * (events[:, 1] - y_mid) / (y_max - y_mid) * (x_max - x_mid))
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W)
        return events[valid_events]

    def scale(self, events, factor):
        scale_events = copy.deepcopy(events)
        H, W = self.resolution
        x_min, x_max = scale_events[:, 0].min().item(), scale_events[:, 0].max().item()
        y_min, y_max = scale_events[:, 1].min().item(), scale_events[:, 1].max().item()
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        scale_events[:, 0] = torch.round((scale_events[:, 0] - x_mid) * factor + x_mid)
        scale_events[:, 1] = torch.round((scale_events[:, 1] - y_mid) * factor + y_mid)
        valid_events = (scale_events[:, 0] >= 0) & (scale_events[:, 0] < W) & (scale_events[:, 1] >= 0) & (scale_events[:, 1] < H)
        scale_events = scale_events[valid_events]
        if scale_events.shape[0] == 0:
            return self.scale(events, factor - 0.1)
        return scale_events

    def event_drop(self, events, option=None):
        raw_events = events
        if option is None:
            option = random.randint(0, 3)  # 0: identity, 1: drop_by_time, 2: drop_by_area, 3: random_drop
        if option == 0:  # identity, do nothing
            return events
        elif option == 1:  # drop_by_time
            T = random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
            events = self.drop_by_time(events, ratio=T)
        elif option == 2:  # drop by area
            area_ratio = random.randint(1, 6) / 20.0  # np.random.uniform(0.05, 0.1, 0.15, 0.2, 0.25)
            events = self.drop_by_area(events, area_ratio=area_ratio)
        elif option == 3:  # random drop
            ratio = random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
            events = self.random_drop(events, ratio=ratio)
        if len(events) == 0:  # avoid dropping all the events
            events = raw_events
        return events

    def mixup_with_cam(self, events, labels, pre_mask=False, post_mask=False):
        H, W = self.resolution
        B = int(1 + events[-1, -1].item())
        new_labels = torch.zeros((*labels.shape, 3)).to(events.device)
        mask_area, cam_area = self.rel_cam.get_mask(events, self.relevance_mix, None)
        aug_events = []
        mix_way = ""
        for b in range(B):
            start = time.time()
            events1 = copy.deepcopy(events[events[:, 4] == b, :4])
            mix_idx = b + 1
            if mix_idx == B:
                mix_idx = 0
            events2 = copy.deepcopy(events[events[:, 4] == mix_idx, :4])

            # mask mix
            if pre_mask:
                events1_int = events1.to(torch.int64)
                events2_int = events2.to(torch.int64)
                mask1 = mask_area[b, events1_int[:, 1], events1_int[:, 0]]
                events1 = events1[mask1]
                mask2 = mask_area[mix_idx, events2_int[:, 1], events2_int[:, 0]]
                events2 = events2[mask2]

            W1 = cam_area[b, 2] - cam_area[b, 0]
            W2 = cam_area[mix_idx, 2] - cam_area[mix_idx, 0]
            H1 = cam_area[b, 3] - cam_area[b, 1]
            H2 = cam_area[mix_idx, 3] - cam_area[mix_idx, 1]
            mid_1_x = int((cam_area[b, 0] + cam_area[b, 2]) / 2)
            mid_1_y = int((cam_area[b, 1] + cam_area[b, 3]) / 2)
            mid_2_x = int((cam_area[mix_idx, 0] + cam_area[mix_idx, 2]) / 2)
            mid_2_y = int((cam_area[mix_idx, 1] + cam_area[mix_idx, 3]) / 2)
            x_inter = W1 + W2 - W
            y_inter = H1 + H2 - H
            inter_area = 0


            if x_inter >= 0 and y_inter >= 0:
                mix_way += "{}both > 0 ".format(b)
                if random.random() > 0.5:
                    linear_1_x = 0 - (mid_1_x - int(W1 / 2))
                    linear_1_y = 0 - (mid_1_y - int(H1 / 2))
                    linear_2_x = W - 1 - (mid_2_x + int(W2 / 2))
                    linear_2_y = H - 1 - (mid_2_y + int(H2 / 2))
                else:
                    linear_1_x = 0 - (mid_1_x - int(W1 / 2))
                    linear_1_y = H - 1 - (mid_1_y + int(H1 / 2))
                    linear_2_x = W - 1 - (mid_2_x + int(W2 / 2))
                    linear_2_y = 0 - (mid_2_y - int(H2 / 2))
                inter_area = x_inter * y_inter

            elif x_inter < 0 and y_inter < 0:
                mix_way += "{}x < 0 y < 0 ".format(b)
                W1_delta = W2 - (W - W1) / 2
                if W1_delta < 0:
                    W1_delta = 0
                H1_delta = H2 - (H - H1) / 2
                if H1_delta < 0:
                    H1_delta = 0
                # Assume events 1 to be in left
                x_l_sample = random.randint(int(W1 / 2), W / 2)
                if x_l_sample > W / 2 - W1_delta:
                    y_l_sample = random.randint(int(H1 / 2), int(H / 2 - H1_delta))
                else:
                    y_l_sample = random.randint(int(H1 / 2), int(H - 1 - H1 / 2))
                if x_l_sample > W / 2 - W1_delta:
                    y_r_sample = random.randint(int(math.ceil(y_l_sample + (H1 + H2) / 2)), int(math.ceil(H - H2 / 2)))
                else:
                    y_r_sample = random.randint(int(H2 / 2), int(H - 1 - H2 / 2))
                if abs(y_r_sample - y_l_sample) < (H1 + H2) / 2:
                    x_r_sample = random.randint(int(x_l_sample + (W1 + W2) / 2), int(W - W2 / 2))
                else:
                    x_r_sample = random.randint(int(W2 / 2), int(W - W2 / 2))
                if random.random() < 0.5:
                    y_l_sample = H - 1 - y_l_sample
                    y_r_sample = H - 1 - y_r_sample
                linear_1_x = x_l_sample - mid_1_x
                linear_1_y = y_l_sample - mid_1_y
                linear_2_x = x_r_sample - mid_2_x
                linear_2_y = y_r_sample - mid_2_y

            elif x_inter >= 0:
                mix_way += "{}x > 0 y < 0 ".format(b)
                # Assume events 1 to be in bottom
                x_d_sample = random.randint(int(W1 / 2), int(W - 1 - W1 / 2))
                y_d_sample = random.randint(int(H1 / 2), int(H - 1 - H2 - H1 / 2))
                x_u_sample = random.randint(int(W2 / 2), int(W - 1 - W2 / 2))
                y_u_sample = random.randint(int(y_d_sample + (H1 + H2) / 2), int(H - 1 - H2 / 2))
                linear_1_x = x_d_sample - mid_1_x
                linear_1_y = y_d_sample - mid_1_y
                linear_2_x = x_u_sample - mid_2_x
                linear_2_y = y_u_sample - mid_2_y

            elif y_inter >= 0:
                mix_way += "{}x < 0 y > 0 ".format(b)
                # Assume events 1 to be in left
                x_l_sample = random.randint(int(W1 / 2), int(W - 1 - W2 - W1 / 2))
                y_l_sample = random.randint(int(H1 / 2), int(H - 1 - H1 / 2))
                x_r_sample = random.randint(int(x_l_sample + (W1 + W2) / 2), int(W - 1 - W2 / 2))
                y_r_sample = random.randint(int(H2 / 2), int(H - 1 - H2 / 2))
                linear_1_x = x_l_sample - mid_1_x
                linear_1_y = y_l_sample - mid_1_y
                linear_2_x = x_r_sample - mid_2_x
                linear_2_y = y_r_sample - mid_2_y
            else:
                raise Exception
            events1[:, 0] += linear_1_x
            events1[:, 1] += linear_1_y
            events2[:, 0] += linear_2_x
            events2[:, 1] += linear_2_y

            new_labels[b, 0] = labels[b]
            new_labels[b, 1] = labels[mix_idx]
            new_labels[b, 2] = W1 * H1 / (W1 * H1 + W2 * H2)
            # mask
            if post_mask:
                x2_min = mid_2_x + linear_2_x - W2 / 2
                x2_max = mid_2_x + linear_2_x + W2 / 2
                y2_min = mid_2_y + linear_2_y - H2 / 2
                y2_max = mid_2_y + linear_2_y + H2 / 2
                index2 = (events2[:, 0] > x2_min) & (events2[:, 0] < x2_max) & (
                        events2[:, 1] > y2_min) & (events2[:, 1] < y2_max)
                index1 = torch.logical_not((events1[:, 0] > x2_min) & (events1[:, 0] < x2_max) & (
                        events1[:, 1] > y2_min) & (events1[:, 1] < y2_max))
                events1 = events1[index1]
                events2 = events2[index2]
                new_labels[b, 2] = (W1 * H1 - inter_area) / (W1 * H1 + W2 * H2 - inter_area)
            aug_single_events = torch.cat([events1, events2], dim=0)
            valid_events = (aug_single_events[:, 0] >= 0) & (aug_single_events[:, 0] < W) & (
                    aug_single_events[:, 1] >= 0) & (aug_single_events[:, 1] < H)
            aug_single_events = aug_single_events[valid_events]
            if aug_single_events.shape[0] == 0:
                aug_single_events = copy.deepcopy(events[events[:, 4] == b, :4]) # raw events
                new_labels[b, 2] = 1
            assert aug_single_events.shape[0] > 0, "rpg mask returns zero event stream"
            aug_single_events = torch.cat(
                [aug_single_events,
                 b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
            aug_events.append(aug_single_events)
        aug_events = torch.cat(aug_events, 0)
        return aug_events, new_labels

    def cut_mix(self, events, labels):
        H, W = self.resolution
        B = int(1 + events[-1, -1].item())
        new_labels = torch.zeros((*labels.shape, 3)).to(self.device)
        aug_events = []
        for b in range(B):
            events1 = copy.deepcopy(events[events[:, 4] == b, :4])
            mix_idx = b + 1
            if mix_idx == B:
                mix_idx = 0
            events2 = copy.deepcopy(events[events[:, 4] == mix_idx, :4])
            ratio = random.random() * 0.9 + 0.1
            H_cut = int(ratio * H)
            W_cut = int(ratio * W)
            valid_region = [events2[:, 0].min(), events2[:, 1].min(), events2[:, 0].max() - W_cut, events2[:, 1].max() - H_cut]
            x_sample_min = random.random() * (valid_region[2] - valid_region[0]) + valid_region[0]
            y_sample_min = random.random() * (valid_region[3] - valid_region[1]) + valid_region[1]
            x_sample_max = x_sample_min + W_cut
            y_sample_max = y_sample_min + H_cut
            index2 = (events2[:, 0] > x_sample_min) & (events2[:, 0] < x_sample_max) & (events2[:, 1] > y_sample_min) & (events2[:, 1] < y_sample_max)
            while len(events2[index2]) == 0:
                x_sample_min = random.random() * (valid_region[2] - valid_region[0]) + valid_region[0]
                y_sample_min = random.random() * (valid_region[3] - valid_region[1]) + valid_region[1]
                x_sample_max = x_sample_min + W_cut
                y_sample_max = y_sample_min + H_cut
                index2 = (events2[:, 0] > x_sample_min) & (events2[:, 0] < x_sample_max) & (
                            events2[:, 1] > y_sample_min) & (events2[:, 1] < y_sample_max)
            events2 = events2[index2]
            index1 = (events1[:, 0] > x_sample_min) & (events1[:, 0] < x_sample_max) & (events1[:, 1] > y_sample_min) & (events1[:, 1] < y_sample_max)
            index1 = torch.logical_not(index1)
            events1 = events1[index1]
            aug_single_events = torch.cat([events1, events2], dim=0)
            valid_events = (aug_single_events[:, 0] >= 0) & (aug_single_events[:, 0] < W) & (
                    aug_single_events[:, 1] >= 0) & (aug_single_events[:, 1] < H)
            aug_single_events = aug_single_events[valid_events]
            aug_single_events = torch.cat(
                [aug_single_events, b * torch.ones((len(aug_single_events), 1), dtype=torch.float32).to(aug_single_events.device)], 1)
            aug_events.append(aug_single_events)
            new_labels[b, 0] = labels[b]
            new_labels[b, 1] = labels[mix_idx]
            new_labels[b, 2] = 1 - ratio ** 2
        aug_events = torch.cat(aug_events, 0)
        return aug_events, new_labels

    def nda(self, events):
        raw_events = copy.deepcopy(events)
        option = random.randint(0, 5)
        if option == 0:  # identity, do nothing
            return events
        elif option == 1:
            events = self.flip_along_x(events, 0)
        elif option == 2:
            linear = 0.6 * (random.random() - 0.5)
            if random.random() < 0.5:
                events = self.linear_x(events, linear)
            else:
                events = self.linear_y(events, linear)
        elif option == 3:
            theta = (random.random() - 0.5) * math.pi
            events = self.rotate(events, theta)
        elif option == 4:
            area_ratio = random.randint(1, 6) / 20.0
            events = self.drop_by_area(events, area_ratio=area_ratio)
        elif option == 5:
            shear = random.random() - 0.5
            events = self.shear_x(events, shear)
        if len(events) == 0:
            events = raw_events
        return events

    def nda_for_rpg(self, events):
        raw_events = copy.deepcopy(events)
        option = random.randint(0, 5)
        if option == 0:  # identity, do nothing
            return events
        elif option == 1:
            events = self.flip_along_x(events, 0)
        elif option == 2:
            linear = 0.6 * (random.random() - 0.5)
            if random.random() < 0.5:
                events = self.linear_x(events, linear)
            else:
                events = self.linear_y(events, linear)
        elif option == 3:
            theta = (random.random() - 0.5) * math.pi
            events = self.rotate(events, theta)
        elif option == 4:
            area_ratio = random.randint(1, 6) / 20.0
            events = self.drop_by_area(events, area_ratio=area_ratio)
        elif option == 5:
            shear = random.random() - 0.5
            events = self.shear_x(events, shear)
        if len(events) == 0:
            events = raw_events
        return events