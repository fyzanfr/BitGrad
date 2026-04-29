import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from bitnet.engine import Tensor
from bitnet.nn import BitLinear, ReLU, Sequential
from bitnet.optim import SGD

# XOR Dataset
X = Tensor([[0,0], [0,1], [1,0], [1,1]])
Y = Tensor([[0], [1], [1], [0]])

# Build a Wide Ternary Model
model = Sequential(
    BitLinear(2, 16),
    ReLU(),
    BitLinear(16, 1)
)

optimizer = SGD(model.parameters(), lr=0.5, weight_decay=0.01)

for epoch in range(300):
    # Forward
    pred = model(X)
    
    # MSE Loss: mean((pred - Y)**2)
    diff = pred - Y
    loss = (diff**2).mean()
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data:.4f}")

final_preds = model(X).data
print("\n--- Final XOR Results ---")
for i in range(4):
    p = 1 if final_preds[i] > 0.5 else 0
    print(f"In: {X.data[i]} -> Target: {Y.data[i]} | Predicted: {p} ({final_preds[i][0]:.4f})")
