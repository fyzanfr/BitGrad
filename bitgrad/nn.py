import numpy as np
from .engine import Tensor
from . import ops

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale)
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class BitLinear(Linear):
    """Ternary Weight Layer {-1, 0, 1}."""
    def forward(self, x):
        w_bit = ops.ternarize(self.weight)
        return x @ w_bit.T + self.bias

class ReLU(Module):
    def forward(self, x): return ops.relu(x)

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
