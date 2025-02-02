import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """
    Mish activation function.
    Method:
        - Applies the mish function element-wise:
        - mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
