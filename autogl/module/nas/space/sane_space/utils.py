import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .operations import *
from torch.autograd import Variable

NA_PRIMITIVES = [
  'sage',
  'sage_sum',
  'sage_max',
  'gcn',
  'gin',
  'gat',
  'gat_sym',
  'gat_cos',
  'gat_linear',
  'gat_generalized_linear',
  'geniepath',
]


SC_PRIMITIVES = [
  'none',
  'skip',
]

LA_PRIMITIVES = [
  'l_max',
  'l_concat',
  'l_lstm',
  # 'l_sum',
  # 'l_att',
  # 'l_mean'
]


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.with_linear = with_linear

    for primitive in NA_PRIMITIVES:
      op = NA_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

  def forward(self, x, weights, edge_index, ):
    mixed_res = []
    if self.with_linear:
      for w, op, linear in zip(weights, self._ops, self._ops_linear):
        mixed_res.append(w * F.elu(op(x, edge_index)+linear(x)))
    else:
      for w, op in zip(weights, self._ops):
        mixed_res.append(w * F.elu(op(x, edge_index)))
    return sum(mixed_res)


class ScMixedOp(nn.Module):

  def __init__(self):
    super(ScMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SC_PRIMITIVES:
      op = SC_OPS[primitive]()
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * op(x))
    return sum(mixed_res)


class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * F.relu(op(x)))
    return sum(mixed_res)


class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.op_linear = nn.Linear(in_dim, out_dim)
    self.act = act_map(act)
    self.with_linear = with_linear

  def forward(self, x, edge_index):
    if self.with_linear:
      return self.act(self._op(x, edge_index)+self.op_linear(x))
    else:
      return self.act(self._op(x, edge_index))


class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)


class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))



