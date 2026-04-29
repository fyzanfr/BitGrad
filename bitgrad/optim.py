import numpy as np

class SGD:
    def __init__(self, params, lr=0.1, max_norm=1.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.max_norm = max_norm
        self.wd = weight_decay

    def step(self):
        # 1. Gradient Clipping (Global Norm)
        total_norm = np.sqrt(sum((p.grad**2).sum() for p in self.params))
        clip_coef = self.max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for p in self.params:
                p.grad *= clip_coef

        # 2. Parameter Update
        for p in self.params:
            # Apply Weight Decay + Gradient Update
            p.data -= self.lr * (p.grad + self.wd * p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)
