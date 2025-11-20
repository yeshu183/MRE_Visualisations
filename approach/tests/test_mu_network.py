"""Test if the mu network can express spatial variation."""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import StiffnessGenerator

print("="*70)
print("MU NETWORK CAPABILITY TEST")
print("="*70)

# Create network
net = StiffnessGenerator(input_dim=1, hidden_dim=64)

# Test with spatial points
x = torch.linspace(0, 1, 100).reshape(-1, 1)

print("\n1. Testing untrained network...")
with torch.no_grad():
    mu_init = net(x)
    
print(f"   Input range:  [{x.min():.3f}, {x.max():.3f}]")
print(f"   Output range: [{mu_init.min():.3f}, {mu_init.max():.3f}]")
print(f"   Output std:   {mu_init.std():.6f}")
print(f"   Unique values: {torch.unique(mu_init).shape[0]}")

if mu_init.std() < 0.001:
    print("   ❌ Network outputs nearly constant values!")
else:
    print("   ✅ Network can express variation")

print("\n2. Testing if network can learn to fit a Gaussian...")
# Target: simple Gaussian
mu_target = 1.0 + 0.5 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))

# Simple gradient descent
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for i in range(200):
    optimizer.zero_grad()
    mu_pred = net(x)
    loss = torch.mean((mu_pred - mu_target) ** 2)
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"   Iter {i:3d}: loss={loss.item():.6f}, mu=[{mu_pred.min():.3f}, {mu_pred.max():.3f}]")

print("\n3. Final result:")
with torch.no_grad():
    mu_final = net(x)
    final_loss = torch.mean((mu_final - mu_target) ** 2).item()
    
print(f"   Target range: [{mu_target.min():.3f}, {mu_target.max():.3f}]")
print(f"   Pred range:   [{mu_final.min():.3f}, {mu_final.max():.3f}]")
print(f"   Final MSE:    {final_loss:.6f}")
print(f"   Output std:   {mu_final.std():.6f}")

if final_loss < 0.01 and mu_final.std() > 0.05:
    print("\n✅ Network CAN learn spatial variation!")
else:
    print("\n❌ Network struggles to learn spatial patterns")

print("="*70)
