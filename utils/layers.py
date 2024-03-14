import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']


def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha=1):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
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


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1):
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)

            S1 = safe_divide(R, Z1)

            C1 = x1 * self.gradprop(Z1, x1, S1)[0]

            return C1

        activator_relevances = f(px)

        out = activator_relevances

        return out


class ZeroPad2d(nn.ZeroPad2d, RelPropSimple):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        outputs = self.X * C[0]
        return outputs


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)



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
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
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


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha=1):
        # X = self.X
        # beta = 1 - alpha
        # weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
        #     (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        # Z = X * weight + 1e-9
        # S = R / Z
        # Ca = S * weight
        # R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
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


class Conv2d(nn.Conv2d, RelProp):
    def relprop(self, R, alpha=1):
        if self.X.grad_fn is None: # input layer
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
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
            return R
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R



