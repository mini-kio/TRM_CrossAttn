import copy
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class EMAHelper(object):
    """
    Exponential Moving Average Helper for model parameters.

    Note: For DDP (DistributedDataParallel), use model.module to access the underlying module.
    DataParallel is deprecated - use DistributedDataParallel instead.
    """
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def _unwrap_module(self, module):
        """Unwrap module from DDP wrapper if necessary"""
        if isinstance(module, DistributedDataParallel):
            return module.module
        return module

    def register(self, module):
        module = self._unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        module = self._unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        module = self._unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

