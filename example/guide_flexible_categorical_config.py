"""
Complete Guide: How to Configure flexible_categorical Hyperparameters
"""
import torch
from prior import flexible_categorical, mlp

print("=" * 70)
print("flexible_categorical Hyperparameter Configuration Guide")
print("=" * 70)

# ============================================================================
# TEMPLATE: Complete hyperparameters dictionary
# ============================================================================

def get_base_hyperparameters():
    """Base hyperparameters template with explanations"""

    hyperparameters = {
        # ----------------------------------------------------------------
        # 1. BASE PRIOR PARAMETERS (for MLP or GP)
        # ----------------------------------------------------------------
        # These are passed to the underlying method (mlp.get_batch or fast_gp.get_batch)

        'num_layers': 3,                      # MLP depth
        'prior_mlp_hidden_dim': 64,           # MLP hidden size
        'is_causal': True,                    # Use causal structure (SCM)
        'num_causes': 10,                     # Number of causal variables
        'noise_std': 0.1,                     # Gaussian noise level
        'y_is_effect': True,                  # Target depends on features
        'pre_sample_weights': True,           # Pre-sample MLP weights
        'prior_mlp_dropout_prob': 0.2,        # Dropout probability
        'pre_sample_causes': True,            # Pre-sample causal structure
        'prior_mlp_activations': lambda: torch.nn.ReLU(),  # Activation function
        'block_wise_dropout': False,
        'prior_mlp_scale_weights_sqrt': True,
        'init_std': 1.0,
        'mix_activations': False,
        'sort_features': False,
        'in_clique': False,
        'random_feature_rotation': False,
        'sampling': 'normal',                 # 'normal', 'uniform', or 'mixed'
        'new_mlp_per_example': False,

        # ----------------------------------------------------------------
        # 2. CLASSIFICATION PARAMETERS
        # ----------------------------------------------------------------
        'num_classes': 3,                     # 0=regression, 2=binary, 3+=multiclass
        'balanced': True,                     # Balance class distribution (binary only)
        'multiclass_type': 'rank',            # 'rank', 'value', or 'multi_node'
        'output_multiclass_ordered_p': 0.5,   # Prob of ordered classes (0.0=unordered)

        # ----------------------------------------------------------------
        # 3. MISSING DATA PARAMETERS
        # ----------------------------------------------------------------
        'nan_prob_no_reason': 0.1,            # Random missing (MCAR) probability
        'nan_prob_a_reason': 0.05,            # Structured missing (MAR) probability
        'nan_prob_unknown_reason': 0.0,       # Unknown mechanism probability
        'nan_prob_unknown_reason_reason_prior': 0.5,  # Mix of above two
        'set_value_to_nan': 0.8,              # 1.0=use NaN, 0.0=use sentinel values

        # ----------------------------------------------------------------
        # 4. CATEGORICAL FEATURES
        # ----------------------------------------------------------------
        'categorical_feature_p': 0.3,         # Probability of categorical features

        # ----------------------------------------------------------------
        # 5. DATA PREPROCESSING
        # ----------------------------------------------------------------
        'normalize_to_ranking': False,        # Use rank transform (True) or z-score (False)

        # ----------------------------------------------------------------
        # 6. FEATURE SELECTION
        # ----------------------------------------------------------------
        'normalize_by_used_features': True,   # Normalize by num_features_used
        'num_features_used': 15,              # Number of informative features
        'seq_len_used': 80,                   # Number of samples used

        # ----------------------------------------------------------------
        # 7. LABEL PROCESSING
        # ----------------------------------------------------------------
        'normalize_labels': True,             # Remap labels to 0,1,2,...
        'check_is_compatible': False,         # Ensure train/test class overlap
        'normalize_ignore_label_too': False,  # Include -100 labels in normalization
        'rotate_normalized_labels': True,     # Randomly rotate class labels
    }

    return hyperparameters


# ============================================================================
# EXAMPLE 1: Binary Classification with Missing Data
# ============================================================================

print("\n" + "=" * 70)
print("Example 1: Binary Classification with Missing Data")
print("=" * 70)

config_binary = {
    # Base MLP
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': False,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.Tanh(),
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False,

    # Binary classification with balanced classes
    'num_classes': 2,
    'balanced': True,  # 50/50 split
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,

    # 20% random missing data
    'nan_prob_no_reason': 0.2,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 1.0,  # Use actual NaN

    # No categorical features
    'categorical_feature_p': 0.0,

    'normalize_to_ranking': False,
    'normalize_by_used_features': False,
    'normalize_labels': True,
    'check_is_compatible': False,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x1, y1, _ = flexible_categorical.get_batch(
    batch_size=8,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=config_binary
)

print(f"\nOutput: x{x1.shape}, y{y1.shape}")
print(f"Classes: {torch.unique(y1).tolist()}")
print(f"Missing: {torch.isnan(x1).sum().item()}/{x1.numel()} ({100*torch.isnan(x1).float().mean():.1f}%)")


# ============================================================================
# EXAMPLE 2: Multi-class Classification with Categorical Features
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: 4-Class Classification with Categorical Features")
print("=" * 70)

config_multiclass = {
    # Base MLP
    'num_layers': 3,
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 8,
    'noise_std': 0.05,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.15,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.ReLU(),
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False,

    # 4-class classification
    'num_classes': 4,
    'balanced': False,  # Not available for >2 classes
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.3,  # 30% ordered, 70% unordered

    # Minimal missing data
    'nan_prob_no_reason': 0.05,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 1.0,

    # 40% of features are categorical
    'categorical_feature_p': 0.4,

    'normalize_to_ranking': False,
    'normalize_by_used_features': True,
    'num_features_used': 15,  # 15 out of 20 are informative
    'seq_len_used': 100,
    'normalize_labels': True,
    'check_is_compatible': False,
    'rotate_normalized_labels': True,
}

x2, y2, _ = flexible_categorical.get_batch(
    batch_size=8,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=config_multiclass,
    single_eval_pos=80
)

print(f"\nOutput: x{x2.shape}, y{y2.shape}")
print(f"Classes: {torch.unique(y2).tolist()}")
print(f"Missing: {torch.isnan(x2).sum().item()}/{x2.numel()} ({100*torch.isnan(x2).float().mean():.1f}%)")


# ============================================================================
# EXAMPLE 3: Regression (Normalized)
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Regression (num_classes=0)")
print("=" * 70)

config_regression = {
    # Base MLP
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.2,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.LeakyReLU(),
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'uniform',
    'new_mlp_per_example': False,

    # Regression
    'num_classes': 0,  # 0 means regression
    'balanced': False,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,

    # No missing data
    'nan_prob_no_reason': 0.0,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 1.0,

    # No categorical features
    'categorical_feature_p': 0.0,

    'normalize_to_ranking': False,
    'normalize_by_used_features': False,
    'normalize_labels': False,
    'check_is_compatible': False,
}

x3, y3, _ = flexible_categorical.get_batch(
    batch_size=8,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=config_regression
)

print(f"\nOutput: x{x3.shape}, y{y3.shape}")
print(f"Target range: [{y3.min():.3f}, {y3.max():.3f}]")
print(f"Target mean: {y3.mean():.3f}")


# ============================================================================
# EXAMPLE 4: High Missing Rate + Categorical
# ============================================================================

print("\n" + "=" * 70)
print("Example 4: Challenging Dataset (High Missing + Categorical)")
print("=" * 70)

config_challenging = {
    # Base MLP
    'num_layers': 3,
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 12,
    'noise_std': 0.15,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.2,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.Tanh(),
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'mixed',
    'new_mlp_per_example': False,

    # Binary classification
    'num_classes': 2,
    'balanced': True,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,

    # HIGH missing data (30% MCAR + 10% MAR)
    'nan_prob_no_reason': 0.3,
    'nan_prob_a_reason': 0.1,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 0.7,  # 70% actual NaN, 30% sentinel values

    # 50% features are categorical
    'categorical_feature_p': 0.5,

    'normalize_to_ranking': False,
    'normalize_by_used_features': True,
    'num_features_used': 12,
    'seq_len_used': 100,
    'normalize_labels': True,
    'check_is_compatible': False,
    'rotate_normalized_labels': True,
}

x4, y4, _ = flexible_categorical.get_batch(
    batch_size=8,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=config_challenging,
    single_eval_pos=80
)

print(f"\nOutput: x{x4.shape}, y{y4.shape}")
print(f"Classes: {torch.unique(y4).tolist()}")
print(f"Missing: {torch.isnan(x4).sum().item()}/{x4.numel()} ({100*torch.isnan(x4).float().mean():.1f}%)")


# ============================================================================
# SUMMARY OF KEY PARAMETERS
# ============================================================================

print("\n" + "=" * 70)
print("QUICK REFERENCE: Key Parameters")
print("=" * 70)

summary = """
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION                                                  │
├─────────────────────────────────────────────────────────────────┤
│ num_classes: 0              → Regression                        │
│ num_classes: 2              → Binary classification             │
│ num_classes: 3+             → Multi-class classification        │
│ balanced: True              → 50/50 split (binary only)         │
│ multiclass_type: 'rank'     → Rank-based classes                │
│ multiclass_type: 'value'    → Fixed threshold classes           │
├─────────────────────────────────────────────────────────────────┤
│ MISSING DATA                                                    │
├─────────────────────────────────────────────────────────────────┤
│ nan_prob_no_reason: 0.2     → 20% random missing (MCAR)         │
│ nan_prob_a_reason: 0.1      → 10% structured missing (MAR)      │
│ set_value_to_nan: 1.0       → Use actual NaN                    │
│ set_value_to_nan: 0.0       → Use sentinel values (-999,0,1,999)│
├─────────────────────────────────────────────────────────────────┤
│ CATEGORICAL FEATURES                                            │
├─────────────────────────────────────────────────────────────────┤
│ categorical_feature_p: 0.3  → 30% features are categorical      │
├─────────────────────────────────────────────────────────────────┤
│ FEATURE SELECTION                                               │
├─────────────────────────────────────────────────────────────────┤
│ num_features_used: 15       → 15 informative features           │
│ (if num_features=20)        → 5 features are zero-padded        │
│ normalize_by_used_features  → Scale by informative ratio        │
├─────────────────────────────────────────────────────────────────┤
│ PREPROCESSING                                                   │
├─────────────────────────────────────────────────────────────────┤
│ normalize_to_ranking: False → Z-score normalization             │
│ normalize_to_ranking: True  → Rank transformation               │
│ normalize_labels: True      → Remap labels to 0,1,2,...         │
│ rotate_normalized_labels    → Randomly shift labels             │
└─────────────────────────────────────────────────────────────────┘
"""

print(summary)

print("\n✅ Configuration guide completed!\n")
