"""
Test script for MLP prior - Basic synthetic data generation
"""
import torch
from prior import mlp

print("=" * 60)
print("Test 1: MLP Prior - Regression Data")
print("=" * 60)

# Configuration for MLP-based synthetic data
hyperparameters = {
    'num_layers': 3,
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.2,
    'pre_sample_causes': True,
    'prior_mlp_activations': 'relu',
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False
}

# Generate batch
batch_size = 8
seq_len = 50  # number of samples
num_features = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nGenerating data on device: {device}")
print(f"Batch size: {batch_size}")
print(f"Sequence length: {seq_len}")
print(f"Number of features: {num_features}")

x, y, y_clean = mlp.get_batch(
    batch_size=batch_size,
    seq_len=seq_len,
    num_features=num_features,
    hyperparameters=hyperparameters,
    device=device,
    num_outputs=1
)

print(f"\n✓ Features shape: {x.shape}")  # (seq_len, batch_size, num_features)
print(f"✓ Targets shape: {y.shape}")     # (seq_len, batch_size)
print(f"\nData statistics:")
print(f"  Features range: [{x.min():.3f}, {x.max():.3f}]")
print(f"  Features mean: {x.mean():.3f}")
print(f"  Targets range: [{y.min():.3f}, {y.max():.3f}]")
print(f"  Targets mean: {y.mean():.3f}")

# Show a sample from the first batch
print(f"\nSample from first batch (first 5 rows):")
print(f"  Features (first 3 dims): \n{x[:5, 0, :3]}")
print(f"  Targets: {y[:5, 0]}")

print("\n✅ MLP prior test completed successfully!\n")
