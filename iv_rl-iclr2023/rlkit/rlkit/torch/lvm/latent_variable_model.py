import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.rlkit.torch.pytorch_util as ptu
from rlkit.rlkit.policies.base import ExplorationPolicy
from rlkit.rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.rlkit.torch.networks import Mlp, CNN
from rlkit.rlkit.torch.networks.basic import MultiInputSequential
from rlkit.rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class LatentVariableModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
