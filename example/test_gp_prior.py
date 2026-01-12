"""
Test script for Gaussian Process prior - Smooth function generation
"""
import torch
from prior import fast_gp

print("=" * 60)
print("Test 2: Gaussian Process Prior - Smooth Functions")
print("=" * 60)

# GP hyperparameters
hyperparameters = {
    'noise': 0.1,
    'outputscale': 1.0,
    'lengthscale': 0.5,
    'sampling': 'uniform',
    'fast_computations': (True, True, True)
}

# Generate smooth functions using Gaussian Process
batch_size = 8
seq_len = 50
num_features = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nGenerating GP data on device: {device}")
print(f"Batch size: {batch_size}")
print(f"Sequence length: {seq_len}")
print(f"Number of features: {num_features}")
print(f"Lengthscale: {hyperparameters['lengthscale']}")

x, y, y_clean = fast_gp.get_batch(
    batch_size=batch_size,
    seq_len=seq_len,
    num_features=num_features,
    device=device,
    hyperparameters=hyperparameters
)

print(f"\n✓ GP Features shape: {x.shape}")
print(f"✓ GP Targets shape: {y.shape}")
print(f"\nData statistics:")
print(f"  Features range: [{x.min():.3f}, {x.max():.3f}]")
print(f"  Features mean: {x.mean():.3f}")
print(f"  Targets range: [{y.min():.3f}, {y.max():.3f}]")
print(f"  Targets mean: {y.mean():.3f}")

# Show how smooth the function is (consecutive points should be similar)
print(f"\nSmoothness check (first batch, consecutive targets):")
for i in range(5):
    print(f"  y[{i}] = {y[i, 0].item():.3f}, y[{i+1}] = {y[i+1, 0].item():.3f}, diff = {abs(y[i, 0] - y[i+1, 0]).item():.3f}")

print("\n✅ GP prior test completed successfully!\n")
