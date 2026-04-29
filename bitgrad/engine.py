import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"


    def __add__(self, other): from . import ops; return ops.add(self, other)
    def __mul__(self, other): from . import ops; return ops.mul(self, other)
    def __matmul__(self, other): from . import ops; return ops.matmul(self, other)
    def __pow__(self, other): from . import ops; return ops.pow_op(self, other)
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    @property
    def T(self): from . import ops; return ops.transpose(self)

    def sum(self): from . import ops; return ops.sum_op(self)

    def mean(self):
        batch_size = np.prod(self.data.shape)
        return self.sum() * (1.0 / batch_size)

    def backward(self):
        # Topological Sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
