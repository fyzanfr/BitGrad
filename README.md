bitgrad 📉

A minimalist, pure-NumPy autograd engine for 1.58-bit (Ternary) Neural Networks.🚀 Core SpecsWeights: Restricted to $\{-1, 0, 1\}$.Engine: Custom micro-autograd with Straight-Through Estimator (STE).Zero Deps: Just Python and NumPy. Optimized for CPU.Stable: Built-in gradient clipping and stable softmax to prevent nan.🛠️ 

Quick StartPython

from bitgrad.nn import BitLinear, Sequential, ReLU
from bitgrad.optim import SGD

model = Sequential(
    BitLinear(2, 16),
    ReLU(),
    BitLinear(16, 1)
)
optimizer = SGD(model.parameters(), lr=0.5)

# Train like any other model—just lighter.
🧩 Structure engine.py: 
Dynamic graph & Tensor class.ops.py: Ternary math & STE logic.nn.py: BitLinear & activations.optim.py: SGD with weight decay.
