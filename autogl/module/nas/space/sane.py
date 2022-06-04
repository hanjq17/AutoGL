import torch
import typing as _typ
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_space
from .base import BaseSpace, map_nn
from ...model import BaseAutoModel
from torch.autograd import Variable
from .operation import act_map
from ..utils import count_parameters, measure_latency

from ..backend import *

from operator import *
from .operation import *
from .sane_space import *
from torch_geometric.utils import add_self_loops

@register_nas_space("sanespace")
class SANENodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 32,
        layer_number: _typ.Optional[int] = 3,
        dropout: _typ.Optional[float] = 0.5,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        search_act_con=False,
    ):
        super().__init__()
        self.ops = ops
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.dropout = dropout
        self.explore_num = 0
        self.with_linear = False
        self.fix_last = True

    def forward(self, data, discrete=False):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index)

        na_weights = F.softmax(self.na_alphas, dim=-1)
        sc_weights = F.softmax(self.sc_alphas, dim=-1)
        la_weights = F.softmax(self.la_alphas, dim=-1)

        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        jk = []
        for i in range(self.layer_number):
            x = self.layers[i](x, na_weights[i], edge_index)  # fix a bug here
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.fix_last and i == self.layer_number - 1:
                jk += [x]
            else:
                jk += [self.scops[i](x, sc_weights[i])]

        merge_feature = self.laop(jk, la_weights[0])
        merge_feature = F.dropout(merge_feature, p=self.dropout, training=self.training)
        logits = self.classifier(merge_feature)
        return F.log_softmax(logits, dim=1)

    def _initialize_alphas(self):
        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)

        self.na_alphas = Variable(1e-3 * torch.randn(self.layer_number, num_na_ops).cuda(), requires_grad=True)
        if self.fix_last:
            self.sc_alphas = Variable(1e-3 * torch.randn(self.layer_number - 1, num_sc_ops).cuda(), requires_grad=True)
        else:
            self.sc_alphas = Variable(1e-3 * torch.randn(self.layer_number, num_sc_ops).cuda(), requires_grad=True)

        self.la_alphas = Variable(1e-3 * torch.randn(1, num_la_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(),
                      F.softmax(self.la_alphas, dim=-1).data.cpu())

        return gene  # this is actually the selection (a string)

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        dropout=None,
    ):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.dropout = dropout or self.dropout

        # TODO: rewrite as setLayerChoice and setInputChoice
        # node aggregator op
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(self.layer_number):
            self.layers.append(NaMixedOp(self.hidden_dim, self.hidden_dim, self.with_linear))

        # skip op
        self.scops = nn.ModuleList()
        for i in range(self.layer_number - 1):
            self.scops.append(ScMixedOp())
        if not self.fix_last:
            self.scops.append(ScMixedOp())

        # layer aggregator op
        self.laop = LaMixedOp(self.hidden_dim, self.layer_number)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim))

        self.model_parameters = [param for name, param in self.named_parameters()]
        self._initialize_alphas()

    def parse_model(self, selection, device) -> BaseAutoModel:
        # sel_list = ['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7]
        temp = lambda dropout, hidden_dim: NetworkGNN(
            selection,
            self.input_dim,
            self.output_dim,
            hidden_dim,
            self.layer_number,
            dropout,
            'relu',
            fix_last=self.fix_last,
            with_linear=self.with_linear
        ).wrap()
        return temp
        # model = NetworkGNN(
        #     selection,
        #     self.input_dim,
        #     self.output_dim,
        #     self.hidden_dim,
        #     self.layer_number,
        #     self.dropout,
        #     'relu',
        #     fix_last=self.fix_last,
        #     with_linear=self.with_linear
        # ).wrap()
        # return model


class NetworkGNN(BaseSpace):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, genotype, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, act='relu', fix_last=False, with_linear=False):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        ops = genotype.split('||')
        self.fix_last = fix_last
        self.with_linear = with_linear

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=self.with_linear) for i in range(num_layers)])

        # skip op
        if self.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList([ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers: 2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                # print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])

        # layer aggregator op
        self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index)

        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        js = []
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == self.num_layers - 1 and self.fix_last:
                js.append(x)
            else:
                js.append(self.sc_layers[i](x))
        x5 = self.layer6(js)
        x5 = F.dropout(x5, p=self.dropout, training=self.training)

        logits = self.classifier(x5)
        return F.log_softmax(logits, dim=1)

