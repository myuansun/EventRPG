import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['sltrp_forward_hook', 'slrp_forward_hook', 'BatchNorm2d', 'Linear', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'Layer',
           'SeqToANNContainer']

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
            px = x.view(-1, self.T, *x_shp).clamp(min=0).mean(1)
            px.requires_grad = True
            self.pX.append(px)
            nx = x.view(-1, self.T, *x_shp).clamp(max=0).mean(1)
            nx.requires_grad = True
            self.nX.append(nx)
            x = x.view(-1, self.T, *x_shp).mean(1)
            x.requires_grad = True
            self.X.append(x)
    else:
        x_shp = input[0].shape[1:]
        self.pX = input[0].detach().view(-1, self.T, *x_shp).clamp(min=0).mean(1)
        self.nX = input[0].detach().view(-1, self.T, *x_shp).clamp(max=0).mean(1)
        self.X = input[0].detach().view(-1, self.T, *x_shp).mean(1)
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

        def f(x):

            Z = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                              ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad, divisor_override=self.divisor_override)
            S = safe_divide(R, Z)
            C = x * self.gradprop(Z, x, S)[0]
            return C

        activator_relevances = f(px)
        inhibitor_relevances = f(nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(RelProp):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, T=10, relprop_mode='sltrp'):
        self.T = T
        if relprop_mode == 'sltrp':
            self.relprop_mode = relprop_mode
        elif relprop_mode == 'slrp':
            self.relprop_mode = None
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.relprop_mode = relprop_mode

    def forward(self, x):
        mem = 0
        total_mem = []
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
            total_mem.append(mem.unsqueeze(1))
        self.mem = torch.cat(total_mem, dim=1)
        return torch.stack(spike_pot, dim=1)

    def relprop(self, R, alpha=1):
        if self.relprop_mode == 'slrp':
            return R
        elif self.relprop_mode == 'sltrp':
            beta = alpha - 1
            X = self.X
            X = torch.clamp(X, min=0)
            pos_X_pre = self.mem[:, 0: -1].clamp(min=0)
            pos_X_post = X[:, 1:].clamp(min=0)
            neg_X_pre = self.mem[:, 0: -1].clamp(max=0)
            neg_X_post = X[:, 1:].clamp(max=0)
            def cal_R_new(R, X_pre, X_post):
                R_new = torch.zeros_like(R)
                weight = safe_divide(X_pre * self.tau, X_post + X_pre * self.tau)
                R_new[:, -1] = R[:, -1]
                for i in reversed(range(weight.shape[1])):
                    R_new[:, i] = R_new[:, i + 1].detach() * weight[:, i] + R[:, i]
                    R_new[:, i + 1] = R_new[:, i + 1].detach() * (1 - weight[:, i])
                return R_new

            activator_relevances = cal_R_new(R, pos_X_pre, pos_X_post)
            inhibitor_relevances = cal_R_new(R, neg_X_pre, neg_X_post)
            R = alpha * activator_relevances - beta * inhibitor_relevances
            return R
        else:
            raise Exception('relevance propagation mode must be sltrp or slrp')


class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, T=10, relprop_mode='sltrp'):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            Conv2d(in_plane, out_plane, kernel_size, stride, padding, T=T, relprop_mode=relprop_mode),
            BatchNorm2d(out_plane, T=T, relprop_mode=relprop_mode)
        )
        self.act = LIFSpike(T=T, relprop_mode=relprop_mode)

    def relprop(self, R, alpha=1):
        R = self.act.relprop(R, alpha)
        R = self.fwd.relprop(R, alpha)
        return R

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = Sequential(*args)

    def relprop(self, R, alpha=1):
        x_shape = [R.shape[0], R.shape[1]]
        R = R.flatten(0, 1)
        R = self.module.relprop(R, alpha)
        x_shape.extend(R.shape[1:])
        R = R.view(x_shape)
        return R

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1))
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class Sequential(nn.Sequential):
    def relprop(self, R, alpha=1):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def __init__(self, *args, T=10, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        nn.BatchNorm2d.__init__(self, *args)

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
        output = super().forward(input)
        return output


class Linear(nn.Linear, RelProp):
    def __init__(self, *args, T=10, relprop_mode='sltrp'):
        self.T = T
        self.relprop_mode = relprop_mode
        super(Linear, self).__init__(*args)

    def relprop(self, R, alpha=1):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        if self.relprop_mode == 'sltrp':
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)
        elif self.relprop_mode == 'slrp':
            px = self.pX
            nx = self.nX
        else:
            raise ValueError('relprop_mode must be sltrp or slrp')

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
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, T=10, relprop_mode="sltrp"):
        self.T = T
        self.relprop_mode = relprop_mode
        nn.Conv2d.__init__(self, in_plane, out_plane, kernel_size, stride, padding)

    def relprop(self, R, alpha=1):
        if self.X.shape[1] == 2 or self.X.shape[1] == 64:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            if self.relprop_mode == 'sltrp':
                px = torch.clamp(self.X, min=0)
                nx = torch.clamp(self.X, max=0)
            elif self.relprop_mode == 'slrp':
                px = self.pX
                nx = self.nX
            else:
                raise ValueError('relprop_mode must be sltrp or slrp')
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
            if self.relprop_mode == 'sltrp':
                px = torch.clamp(self.X, min=0)
                nx = torch.clamp(self.X, max=0)
            elif self.relprop_mode == 'slrp':
                px = self.pX
                nx = self.nX
            else:
                raise ValueError('relprop_mode must be sltrp or slrp')


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
