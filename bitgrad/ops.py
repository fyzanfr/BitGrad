import numpy as np

def _wrap(x):
    """Ensures inputs are Tensors without causing a circular import crash."""
    from .engine import Tensor
    return x if hasattr(x, '_prev') else Tensor(x)

def _unbroadcast(grad, shape):
    """Reshapes gradients for batched operations."""
    res = grad
    while res.ndim > len(shape): res = res.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1: res = res.sum(axis=i, keepdims=True)
    return res

def add(a, b):
    from .engine import Tensor
    b = _wrap(b)
    out = Tensor(a.data + b.data, (a, b), '+')
    def _backward():
        a.grad += _unbroadcast(out.grad, a.data.shape)
        b.grad += _unbroadcast(out.grad, b.data.shape)
    out._backward = _backward
    return out

def mul(a, b):
    from .engine import Tensor
    b = _wrap(b)
    out = Tensor(a.data * b.data, (a, b), '*')
    def _backward():
        a.grad += _unbroadcast(out.grad * b.data, a.data.shape)
        b.grad += _unbroadcast(out.grad * a.data, b.data.shape)
    out._backward = _backward
    return out

def matmul(a, b):
    from .engine import Tensor
    b = _wrap(b)
    out = Tensor(a.data @ b.data, (a, b), 'matmul')
    def _backward():
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad
    out._backward = _backward
    return out

def relu(a):
    from .engine import Tensor
    out = Tensor(np.maximum(0, a.data), (a,), 'relu')
    def _backward():
        a.grad += (a.data > 0).astype(np.float32) * out.grad
    out._backward = _backward
    return out

def ternarize(w):
    """Core BitNet b1.58 STE: maps to {-1, 0, 1} while passing grads."""
    from .engine import Tensor
    gamma = np.mean(np.abs(w.data)) + 1e-7
    w_quant = np.clip(np.round(w.data / gamma), -1, 1) * gamma
    out = Tensor(w_quant, (w,), 'ternarize')
    def _backward():
        # Straight-Through Estimator: gradients bypass the rounding
        w.grad += out.grad 
    out._backward = _backward
    return out

def pow_op(a, n):
    from .engine import Tensor
    out = Tensor(a.data**n, (a,), f'**{n}')
    def _backward():
        a.grad += (n * (a.data**(n-1))) * out.grad
    out._backward = _backward
    return out

def sum_op(a):
    from .engine import Tensor
    out = Tensor(np.sum(a.data), (a,), 'sum')
    def _backward():
        a.grad += np.ones_like(a.data) * out.grad
    out._backward = _backward
    return out

def transpose(a):
    from .engine import Tensor
    out = Tensor(a.data.T, (a,), 'T')
    def _backward():
        a.grad += out.grad.T
    out._backward = _backward
    return out
