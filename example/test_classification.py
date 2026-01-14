"""
Test script for Classification data generation using flexible_categorical
"""
import torch
from prior import flexible_categorical, mlp

print("=" * 60)
print("Test 3: Classification Data with flexible_categorical")
print("=" * 60)

# Classification configuration
hyperparameters = {
    # MLP configuration
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': False,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'prior_mlp_activations': 'tanh',
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False,

    # Classification wrapper config
    'num_classes': 3,  # For 3-class classification
    'balanced': False,  # balanced only works for binary classification
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.5,
    'nan_prob_no_reason': 0.1,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'nan_prob_unknown_reason_reason_prior': 0.5,
    'categorical_feature_p': 0.2,
    'normalize_to_ranking': False,
    'set_value_to_nan': 0.5,
    'normalize_by_used_features': True,
    'num_features_used': 12,
    'seq_len_used': 60,
    'normalize_labels': True,
    'check_is_compatible': False,
    'normalize_ignore_label_too': False,
    'rotate_normalized_labels': True
}

# Generate classification data
batch_size = 16
seq_len = 80
num_features = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nGenerating classification data on device: {device}")
print(f"Batch size: {batch_size}")
print(f"Sequence length: {seq_len}")
print(f"Number of features: {num_features}")
print(f"Number of classes: {hyperparameters['num_classes']}")
print(f"Balanced: {hyperparameters['balanced']}")

x, y, y_clean = flexible_categorical.get_batch(
    batch_size=batch_size,
    seq_len=seq_len,
    num_features=num_features,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=hyperparameters,
    batch_size_per_gp_sample=8,
    single_eval_pos=60
)

print(f"\n✓ Classification features shape: {x.shape}")
print(f"✓ Classification targets shape: {y.shape}")

# Check unique classes
unique_classes = torch.unique(y[~torch.isnan(y)])
print(f"\nUnique classes: {unique_classes.tolist()}")

# Check class distribution
print(f"\nClass distribution (all batches):")
for cls in unique_classes:
    count = (y == cls).sum().item()
    percentage = count / (~torch.isnan(y)).sum().item() * 100
    print(f"  Class {int(cls)}: {count} samples ({percentage:.1f}%)")

# Check for missing values
nan_count = torch.isnan(x).sum().item()
total_values = x.numel()
print(f"\nMissing values: {nan_count} / {total_values} ({nan_count/total_values*100:.1f}%)")

# Show a sample
print(f"\nSample from first batch (first 5 rows):")
print(f"  Features (first 3 dims): \n{x[:5, 0, :3]}")
print(f"  Targets: {y[:5, 0]}")

print("\n✅ Classification test completed successfully!\n")
