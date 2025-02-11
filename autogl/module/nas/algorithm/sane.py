# Modified from NNI

import logging

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from ..utils import replace_layer_choice, replace_input_choice
# from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, DartsInputChoice

_logger = logging.getLogger(__name__)

# copy from nni2.1 for stablility
# class DartsLayerChoice(nn.Module):
#     def __init__(self, layer_choice):
#         super(DartsLayerChoice, self).__init__()
#         self.name = layer_choice.key
#         self.op_choices = nn.ModuleDict(layer_choice.named_children())
#         self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
#
#     def forward(self, *args, **kwargs):
#         op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
#         alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
#         return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)
#
#     def parameters(self):
#         for _, p in self.named_parameters():
#             yield p
#
#     def named_parameters(self):
#         for name, p in super(DartsLayerChoice, self).named_parameters():
#             if name == 'alpha':
#                 continue
#             yield name, p
#
#     def export(self):
#         return torch.argmax(self.alpha).item()
#
#
# class DartsInputChoice(nn.Module):
#     def __init__(self, input_choice):
#         super(DartsInputChoice, self).__init__()
#         self.name = input_choice.key
#         self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
#         self.n_chosen = input_choice.n_chosen or 1
#
#     def forward(self, inputs):
#         inputs = torch.stack(inputs)
#         alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
#         return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)
#
#     def parameters(self):
#         for _, p in self.named_parameters():
#             yield p
#
#     def named_parameters(self):
#         for name, p in super(DartsInputChoice, self).named_parameters():
#             if name == 'alpha':
#                 continue
#             yield name, p
#
#     def export(self):
#         return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]

@register_nas_algo("sane")
class SANE(BaseNAS):
    """
    SANE trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    workers : int
        Workers for data loading.
    gradient_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    model_lr : float
        Learning rate to optimize the model.
    model_wd : float
        Weight decay to optimize the model.
    arch_lr : float
        Learning rate to optimize the architecture.
    arch_wd : float
        Weight decay to optimize the architecture.
    device : str or torch.device
        The device of the whole process
    """

    def __init__(
        self,
        num_epochs=5,
        workers=4,
        gradient_clip=5.0,
        model_lr=1e-3,
        model_wd=5e-4,
        arch_lr=3e-4,
        arch_wd=1e-3,
        device="auto",
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.workers = workers
        self.gradient_clip = gradient_clip
        self.model_optimizer = torch.optim.Adam
        self.arch_optimizer = torch.optim.SGD
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd

    def search(self, space: BaseSpace, dataset, estimator):
        model_optim = self.model_optimizer(
            space.model_parameters, self.model_lr, weight_decay=self.model_wd
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, float(self.num_epochs),
                                                               eta_min=0.001)

        space = space.to(self.device)

        arch_optim = self.arch_optimizer(
            space.arch_parameters(), self.arch_lr, weight_decay=self.arch_wd
        )

        for epoch in range(self.num_epochs):
            self._train_one_epoch(
                epoch, space, dataset, estimator, model_optim, arch_optim
            )
            scheduler.step()

        selection = space.genotype()  # TODO: should be dict, but now is str
        print(selection)
        return space.parse_model(selection, self.device)

    def _train_one_epoch(
        self,
        epoch,
        model: BaseSpace,
        dataset,
        estimator,
        model_optim: torch.optim.Optimizer,
        arch_optim: torch.optim.Optimizer,
    ):
        model.train()

        # phase 1. architecture step
        arch_optim.zero_grad()
        # only no unroll here
        _, loss = self._infer(model, dataset, estimator, "val")
        if epoch % 10 == 0:
            print('Val acc:', _[0].item() * 100)
            print('Val loss:', loss.item())
        loss.backward()
        arch_optim.step()

        # phase 2: child network step
        model_optim.zero_grad()
        metric, loss = self._infer(model, dataset, estimator, "train")
        loss.backward()
        if epoch % 10 == 0:
            print('Train acc:', metric[0].item() * 100)
            print('Train loss:', loss.item())
        # gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        model_optim.step()

    def _infer(self, model: BaseSpace, dataset, estimator: BaseEstimator, mask="train"):
        metric, loss = estimator.infer(model, dataset, mask=mask)
        return metric, loss

