import math
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.common.common import ndarray

from deep_rl import BaseNet, Config, layer_init


class StateNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(StateNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.ones(1, num_features))
        self.batch = deque(maxlen=256)

    def update(self, first_time=False):
        batch = torch.cat(list(self.batch), dim=0)
        self.batch = deque(maxlen=256)

        mean = batch.mean(0, keepdim=True)
        var = batch.var(0, keepdim=True)

        if first_time:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
        else:
            momentum = self.momentum
            self.running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            self.running_var.mul_(1 - momentum).add_(var, alpha=momentum)

    def forward(self, input):
        if self.training:
            self.batch.append(input.clone().detach())
        return input.sub(self.running_mean).div(torch.sqrt(self.running_var + self.eps))


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, norm_type=None):
        super(LinearBlock, self).__init__()
        self.norm_type = norm_type
        self.linear = nn.Linear(in_features, out_features, bias=norm_type is None)
        if norm_type is None:
            self.norm = nn.Identity()
        elif norm_type == nn.LayerNorm:
            self.norm = norm_type(out_features, elementwise_affine=True)
        else:
            self.norm = norm_type(out_features, affine=True)
        self.nonlinearity = nn.Tanh()

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.nonlinearity(out)
        return out


class Controller(nn.Module):
    def __init__(self, deformable):
        super(Controller, self).__init__()
        self.deformable = deformable
        self.dofs = deformable.dofs()
        self.act_dofs = deformable.act_dofs()
        self.all_muscles = deformable.all_muscles


class NNController(Controller):
    def __init__(self, deformable, widths, norm_type=None, dropout=0.0):
        super(NNController, self).__init__(deformable)
        self.layers = nn.ModuleList()
        # state_norm = StateNorm(widths[0])
        # self.layers.append(state_norm)
        for i in range(len(widths) - 1):
            in_feature, out_features = widths[i], widths[i + 1]
            if i < len(widths) - 2:
                self.layers.append(LinearBlock(
                    in_feature, out_features, norm_type))
            else:
                if dropout > 0.0:
                    self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.Linear(widths[i], widths[i + 1], bias=True))

    def get_state_norm(self):
        return self.layers[0]

    def update_state_norm(self, states):
        self.get_state_norm().update(states)

    def reset_parameters(self, gain=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, prev_a):
        raise NotImplementedError


class IndNNController(NNController):
    def __init__(self, deformable, widths, norm_type=None, dropout=0.0):
        super(IndNNController, self).__init__(deformable, widths, norm_type, dropout)
        muscle_dofs = 0
        for shared_muscles in self.all_muscles:
            muscle_dofs += len(shared_muscles[0][0])
        self.layers[-1] = nn.Linear(self.layers[-1].in_features, muscle_dofs, bias=True)

    def forward(self, x, prev_a) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        x = x.tanh().squeeze(0)

        a_shared_muscles = []
        a = []

        pointer = 0
        for shared_muscles in self.all_muscles:
            a_shared_muscles.append(x[pointer:pointer + len(shared_muscles[0][0])])

        for w, shared_muscles in zip(a_shared_muscles, self.all_muscles):
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    a.append(mu)

        a = torch.cat(a)
        return 1 - a


class AdaNNController(NNController):
    def __init__(self, deformable, widths, norm_type=None, dropout=0.0, segment_len=1):
        super(AdaNNController, self).__init__(deformable, widths, norm_type, dropout)
        self.a_init = torch.zeros(self.act_dofs, requires_grad=False)
        self.segment_len = segment_len

    def forward(self, x, prev_a) -> torch.Tensor:
        segment_len = self.segment_len
        if prev_a is None:
            prev_a = self.a_init
        else:
            prev_a = 1 - prev_a
        a = []
        pointer = 0

        for layer in self.layers:
            x = layer(x)
        x = x.tanh().squeeze(0)

        for w, shared_muscles in zip(x, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                    pointer += len(muscle)
                    a_cord = torch.cat([mu.expand(self.segment_len), prev_a_cord[:-1]])
                    a.append(a_cord)
        a = torch.cat(a)
        return 1 - a


class SnakeAdaNNController(AdaNNController):
    def __init__(self, deformable, widths, norm_type=None, dropout=0.0):
        super(SnakeAdaNNController, self).__init__(deformable, widths, norm_type, dropout)
        self.prev_a_clean = None

    def forward(self, x, prev_a) -> torch.Tensor:
        if prev_a is None:
            self.prev_a_clean = self.a_init
        elif not prev_a.requires_grad:
            self.prev_a_clean.detach_().requires_grad_(False)
        a = []
        a_clean = []
        pointer = 0

        for layer in self.layers:
            x = layer(x)
        x = x.tanh().squeeze(0)

        for w, shared_muscles in zip(x, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')

                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = self.prev_a_clean[pointer:pointer + len(muscle)]
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a_clean.append(a_cord)

                    weight = torch.linspace(0.5, 1.0, len(muscle), requires_grad=False)
                    a_cord = weight * a_cord
                    a.append(a_cord)
                    pointer += len(muscle)
        a = torch.cat(a)
        self.prev_a_clean = torch.cat(a_clean)
        return 1 - a


class OpenController(Controller):
    def __init__(self, deformable, ctrl_num):
        super(OpenController, self).__init__(deformable)
        self.ctrl_num = ctrl_num
        self.weight = nn.Parameter(torch.Tensor(*self.weight_shape()))

    def update_weight(self, weight):
        self.weight.data.copy_(weight.view(*self.weight_shape()))

    def get_grad(self):
        return self.weight.grad

    def weight_shape(self) -> tuple:
        raise NotImplementedError

    def forward(self, step, prev_a):
        raise NotImplementedError


class SharedOpenController(OpenController):
    def weight_shape(self) -> tuple:
        return (self.ctrl_num, len(self.all_muscles))

    def forward(self, step, prev_a):
        a = []
        for w, shared_muscles in zip(self.weight[step], self.all_muscles):
            for muscle_pair in shared_muscles:
                if len(muscle_pair) == 1:
                    a.append(w.sigmoid().expand(len(muscle_pair[0])))
                elif len(muscle_pair) == 2:

                    # map (-1, 1) to either (-1, 0) or (0, 1)
                    w = w.tanh()
                    a.append(0.5 * (w.abs() - w))
                    a.append(0.5 * (w.abs() + w))
                else:
                    raise ValueError(f'invalid # muscle pair: {len(muscle_pair)}')
        a = torch.cat(a)
        return 1 - a


class AdaOpenController(OpenController):
    def __init__(self, deformable, ctrl_num):
        super(AdaOpenController, self).__init__(deformable, ctrl_num)
        self.a_init = torch.zeros(self.act_dofs, requires_grad=False)

    def weight_shape(self) -> tuple:
        return (self.ctrl_num, len(self.all_muscles))

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)

    def forward(self, step, prev_a) -> torch.Tensor:
        if prev_a is None:
            prev_a = self.a_init
        else:
            prev_a = 1 - prev_a
        a = []
        pointer = 0
        weight = self.weight[step].tanh()
        for w, shared_muscles in zip(weight, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                    pointer += len(muscle)
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a.append(a_cord)
        a = torch.cat(a)
        return 1 - a


class SnakeAdaOpenController(AdaOpenController):
    def __init__(self, deformable, ctrl_num):
        super(SnakeAdaOpenController, self).__init__(deformable, ctrl_num)
        self.prev_a_clean = None

    def forward(self, step, prev_a) -> torch.Tensor:
        if prev_a is None:
            self.prev_a_clean = self.a_init
        a = []
        a_clean = []
        pointer = 0
        weight = self.weight[step].tanh()
        for w, shared_muscles in zip(weight, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = self.prev_a_clean[pointer:pointer + len(muscle)]
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a_clean.append(a_cord)

                    weight = torch.linspace(0.5, 1.0, len(muscle), requires_grad=False)
                    a_cord = weight * a_cord
                    a.append(a_cord)
                    pointer += len(muscle)
        a = torch.cat(a)
        self.prev_a_clean = torch.cat(a_clean)
        return 1 - a
