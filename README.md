# bitgrad 📉

A minimalist, pure-NumPy autograd engine for **1.58-bit (Ternary) Neural Networks**.

## 🚀 Core Specs

- **Weights:** Restricted to {-1, 0, 1}.
- **Engine:** Custom micro-autograd with Straight-Through Estimator (STE).
- **Zero Deps:** Just Python and NumPy. Optimized for CPU.
- **Stable:** Built-in gradient clipping and stable softmax to prevent `nan`.

## 🛠️ Quick Start

```python
from bitgrad.nn import BitLinear, Sequential, ReLU
from bitgrad.optim import SGD

# Build a wide Ternary model
model = Sequential(
    BitLinear(2, 16),
    ReLU(),
    BitLinear(16, 1)
)

# High Learning Rate is key for Ternary training
optimizer = SGD(model.parameters(), lr=0.5)
```
# Train like any other model—just lighter.
