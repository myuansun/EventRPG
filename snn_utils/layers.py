import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import layer
import copy

__all__ = ['sltrp_forward_hook', 'slrp_forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d', 'Multiply', 'Mean',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'IFNode', 'ParametricLIFNode', 'LIFNode']

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


def sltrp_forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

def slrp_forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.pX = []
        self.nX = []
        self.X = []
        for i in input[0]:
            x = i.detach()
            x_shp = x.shape[1:]
            px = x.view(self.T, -1, *x_shp).clamp(min=0).mean(0)
            px.requires_grad = True
            self.pX.append(px)
            nx = x.view(self.T, -1, *x_shp).clamp(max=0).mean(0)
            nx.requires_grad = True
            self.nX.append(nx)
            x = x.view(self.T, -1, *x_shp).mean(0)
            x.requires_grad = True
            self.X.append(x)
    else:
        x_shp = input[0].shape[1:]
        self.pX = input[0].detach().view(self.T, -1, *x_shp).clamp(min=0).mean(0)
        self.nX = input[0].detach().view(self.T, -1, *x_shp).clamp(max=0).mean(0)
        self.X = input[0].detach().view(self.T, -1, *x_shp).mean(0)
        self.pX.requires_grad = True
        self.nX.requires_grad = True
        self.X.requires_grad = True


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        if self.relprop_mode == 'sltrp':
            self.register_forward_hook(sltrp_forward_hook)
        elif self.relprop_mode == 'slrp':
            self.register_forward_hook(slrp_forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha=1):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        R = R.view(Z.shape)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs


class Identity(nn.Identity, RelProp):
    pass


class ReLU(nn.ReLU, RelProp):
    pass


class LeakyReLU(nn.LeakyReLU, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(layer.MaxPool2d, RelProp):
    def __init__(self, *args, T=1, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        layer.MaxPool2d.__init__(self, *args)

    def relprop(self, R, alpha=1):
        beta = alpha - 1
        if self.relprop_mode == 'sltrp':
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')

        if self.step_mode == "m":
            x_shape = [self.X.shape[0], self.X.shape[1]]
            px = px.flatten(0, 1)
            nx = nx.flatten(0, 1)
            R = R.flatten(0, 1)

        def f(x):

            Z = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1,
                              ceil_mode=self.ceil_mode, return_indices=self.return_indices)
            S = safe_divide(R, Z)
            C = x * self.gradprop(Z, x, S)[0]
            return C

        activator_relevances = f(px)
        inhibitor_relevances = f(nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances
        if self.step_mode == "m":
            x_shape.extend(R.shape[1:])
            R = R.view(x_shape)
        return R

class Mean(RelPropSimple):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, X):
        return X.mean(self.dim)

class AdaptiveAvgPool2d(layer.AdaptiveAvgPool2d, RelProp):
    def __init__(self, *args, T=16, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        layer.AdaptiveAvgPool2d.__init__(self, *args)

    def relprop(self, R, alpha=1):
        beta = alpha - 1
        if self.relprop_mode == 'sltrp':
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode should be slrp or sltrp')

        def f(x1):
            Z1 = self.forward(x1)

            S1 = safe_divide(R, Z1)

            C1 = x1 * self.gradprop(Z1, x1, S1)[0]

            return C1

        activator_relevances = f(px)
        inhibitor_relevances = f(nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances
        return R


class ZeroPad2d(nn.ZeroPad2d, RelPropSimple):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        outputs = self.X * C[0]
        return outputs


class AvgPool2d(nn.AvgPool2d, RelProp):
    def __init__(self, *args, T=1, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        nn.AvgPool2d.__init__(self, *args)

    def relprop(self, R, alpha=1):
        
        beta = alpha - 1
        if self.relprop_mode == 'sltrp':
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')

        if self.step_mode == "m":
            x_shape = [self.X.shape[0], self.X.shape[1]]
            px = px.flatten(0, 1)
            nx = nx.flatten(0, 1)
            R = R.flatten(0, 1)
        def f(x):

            Z = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                              ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad, divisor_override=self.divisor_override)
            S = safe_divide(R, Z)
            C = x * self.gradprop(Z, x, S)[0]
            return C

        activator_relevances = f(px)
        inhibitor_relevances = f(nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        if self.step_mode == "m":
            x_shape.extend(R.shape[1:])
            R = R.view(x_shape)
        return R


class Add(RelProp):
    def __init__(self, T=16, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        super().__init__()

    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha=1):
        beta = alpha - 1
        if self.relprop_mode == 'sltrp':
            px = list(map(lambda x: torch.clamp(x, min=0), self.X))
            nx = list(map(lambda x: torch.clamp(x, max=0), self.X))
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')

        def f(x):

            Z = torch.add(*x)
            S = safe_divide(R, Z)
            x_grad = self.gradprop(Z, x, S)
            C = list(map(lambda x: x[0] * x[1], zip(x, x_grad)))
            return C

        activator_relevances = f(px)
        inhibitor_relevances = f(nx)

        R = list(map(lambda x: alpha * x[0] - beta * x[1], zip(activator_relevances, inhibitor_relevances)))
        return R


class Multiply(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)

    def relprop(self, R, alpha=1):
        x0 = torch.clamp(self.X[0], min=0)
        x1 = torch.clamp(self.X[1], min=0)
        x = [x0, x1]
        Z = self.forward(x)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, x, S)

        outputs = []
        outputs.append(x[0] * C[0])
        outputs.append(x[1] * C[1])

        return outputs


class Clone(RelProp):
    def __init__(self, T=16, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        super().__init__()

    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha=1):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r.view(z.shape), z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        R = self.X * C
        return R



class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha=1):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha=1):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class BatchNorm2d(layer.BatchNorm2d, RelProp):
    def __init__(self, *args, T=10, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        layer.BatchNorm2d.__init__(self, *args)


    def relprop(self, R, alpha=1):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R

    def forward(self, input):
        output = layer.BatchNorm2d.forward(self, input)
        return output


class Linear(layer.Linear, RelProp):
    def __init__(self, *args, T=10, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        super(Linear, self).__init__(*args)

    def relprop(self, R, alpha=1):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S)[0]
            C2 = x2 * self.gradprop(Z2, x2, S)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        out = alpha * activator_relevances - beta * inhibitor_relevances

        return out


class Conv2d(layer.Conv2d, RelProp):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, dilation=1, groups=1, bias=False, T=16, relprop_mode="sltrp"):
        self.T = T
        self.relprop_mode = relprop_mode
        layer.Conv2d.__init__(self, in_plane, out_plane, kernel_size, stride, padding, dilation, groups, bias)

    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha=1):
        if self.relprop_mode == 'sltrp':
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')

        if self.step_mode == "m":
            x_shape = [self.X.shape[0], self.X.shape[1]]
            px = px.flatten(0, 1)
            nx = nx.flatten(0, 1)
            R = R.flatten(0, 1)

        if px.shape[1] == 2 or px.shape[1] == 64:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=torch.zeros(w1.shape[0]).to(w1.device), stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=torch.zeros(w1.shape[0]).to(w1.device), stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                Rp = torch.clamp(R, min=0)
                Rn = torch.clamp(R, max=0)
                Rp_sum = Rp.sum()
                Rn_sum = Rn.sum()
                mask = (Z != 0).to(torch.float32)
                R_p_mask = Rp * mask
                R_n_mask = Rn * mask
                Rp = safe_divide(R_p_mask * Rp_sum, R_p_mask.sum())
                Rn = safe_divide(R_n_mask * Rn_sum, R_n_mask.sum())
                redis_R = Rp + Rn
                S = safe_divide(redis_R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=torch.zeros(w1.shape[0]).to(w1.device), stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=torch.zeros(w1.shape[0]).to(w1.device), stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        if self.step_mode == "m":
            x_shape.extend(R.shape[1:])
            R = R.view(x_shape)
        return R


class IFNode(neuron.IFNode, RelProp):
    def __init__(self, *args, **kargs):
        super(IFNode, self).__init__(*args, **kargs)
        self.T = kargs['T']
        self.relprop_mode = kargs['relprop_mode']
        self.past_v = torch.tensor([])

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
        self.past_v = torch.cat([self.past_v.to(self.v.device), self.v.unsqueeze(0).detach()])

    def relprop(self, R, alpha=1):
        if self.relprop_mode == 'slrp':
            return R
        elif self.relprop_mode == 'sltrp':
            beta = alpha - 1
            X = self.X
            X = torch.clamp(X, min=0)
            pos_X_pre = self.past_v[0: -1].clamp(min=0)
            pos_X_post = X[1:].clamp(min=0)
            neg_X_pre = self.past_v[0: -1].clamp(max=0)
            neg_X_post = X[1:].clamp(max=0)

            def cal_R_new(R, X_pre, X_post):
                R_new = torch.zeros_like(R)
                weight = safe_divide(X_pre, X_post + X_pre)
                R_new[-1] = R[-1]
                for i in reversed(range(weight.shape[1])):
                    R_new[i] = R_new[i + 1].detach() * weight[i] + R[i]
                    R_new[i + 1] = R_new[i + 1].detach() * (1 - weight[:, i])
                return R_new

            activator_relevances = cal_R_new(R, pos_X_pre, pos_X_post)
            inhibitor_relevances = cal_R_new(R, neg_X_pre, neg_X_post)
            R = alpha * activator_relevances - beta * inhibitor_relevances
            return R
        else:
            raise Exception('relevance propagation mode must be sltrp or slrp')

    def reset(self):
        self.past_v = torch.tensor([])
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])



class ParametricLIFNode(neuron.ParametricLIFNode, RelProp):
    def __init__(self, *args, **kwargs):
        self.T = kwargs['T']
        self.relprop_mode = kwargs['relprop_mode']
        del kwargs['T']
        del kwargs['relprop_mode']
        super(ParametricLIFNode, self).__init__(*args, **kwargs)
        self.past_v = torch.tensor([])

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
        self.past_v = torch.cat([self.past_v.to(self.v.device), self.v.unsqueeze(0).detach()])

    def relprop(self, R, alpha=1):
        if self.relprop_mode == 'slrp':
            return R
        elif self.relprop_mode == 'sltrp':
            beta = alpha - 1
            X = self.X
            X = torch.clamp(X, min=0)
            pos_X_pre = self.past_v[0: -1].clamp(min=0)
            pos_X_post = X[1:].clamp(min=0)
            neg_X_pre = self.past_v[0: -1].clamp(max=0)
            neg_X_post = X[1:].clamp(max=0)

            def cal_R_new(R, X_pre, X_post):
                R_new = torch.zeros_like(R)
                tau = self.w.sigmoid()
                weight = safe_divide(X_pre * (1 - tau), X_post * tau + X_pre * (1 - tau))
                R_new[-1] = R[-1]
                for i in reversed(range(weight.shape[0])):
                    R_new[i] = R_new[i + 1].detach() * weight[i] + R[i]
                    R_new[i + 1] = R_new[i + 1].detach() * (1 - weight[i])
                return R_new

            activator_relevances = cal_R_new(R, pos_X_pre, pos_X_post)
            inhibitor_relevances = cal_R_new(R, neg_X_pre, neg_X_post)
            R = alpha * activator_relevances - beta * inhibitor_relevances
            return R
        else:
            raise Exception('relevance propagation mode must be sltrp or slrp')

    def reset(self):
        self.past_v = torch.tensor([])
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

class LIFNode(neuron.LIFNode, RelProp):
    def __init__(self, *args, **kwargs):
        self.T = kwargs['T']
        self.relprop_mode = kwargs['relprop_mode']
        del kwargs['T']
        del kwargs['relprop_mode']
        super(LIFNode, self).__init__(*args, **kwargs)
        self.past_v = torch.tensor([])

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
        self.past_v = torch.cat([self.past_v.to(self.v.device), self.v.unsqueeze(0).detach()])

    def relprop(self, R, alpha=1):
        if self.relprop_mode == 'slrp':
            return R
        elif self.relprop_mode == 'sltrp':
            beta = alpha - 1
            X = self.X
            X = torch.clamp(X, min=0)
            pos_X_pre = self.past_v[0: -1].clamp(min=0)
            pos_X_post = X[1:].clamp(min=0)
            neg_X_pre = self.past_v[0: -1].clamp(max=0)
            neg_X_post = X[1:].clamp(max=0)

            def cal_R_new(R, X_pre, X_post):
                R_new = torch.zeros_like(R)
                weight = safe_divide(X_pre * self.tau, X_post + X_pre * self.tau)
                R_new[-1] = R[-1]
                for i in reversed(range(weight.shape[1])):
                    R_new[i] = R_new[i + 1].detach() * weight[i] + R[i]
                    R_new[i + 1] = R_new[i + 1].detach() * (1 - weight[:, i])
                return R_new

            activator_relevances = cal_R_new(R, pos_X_pre, pos_X_post)
            inhibitor_relevances = cal_R_new(R, neg_X_pre, neg_X_post)
            R = alpha * activator_relevances - beta * inhibitor_relevances
            return R
        else:
            raise Exception('relevance propagation mode must be sltrp or slrp')

    def reset(self):
        self.past_v = torch.tensor([])
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])
