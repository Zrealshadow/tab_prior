"""
Example: Wrapping flexible_categorical with differentiable_prior

This shows the full pipeline:
  differentiable_prior (samples hyperparameters)
    → flexible_categorical (post-processes data)
      → mlp (generates raw data)

Use this when you want:
- Random hyperparameter sampling (num_layers, noise_std, etc.)
- Classification conversion (regression → binary/multiclass)
- Missing values, categorical features, normalization
"""
from functools import partial

import torch
from prior.differentiable_prior import get_batch as diff_get_batch
from prior import flexible_categorical, mlp


# Hyperparameters for the full pipeline
hyperparameters = {
    # ==========================================================================
    # MLP hyperparameters (some sampled, some fixed)
    # ==========================================================================

    # === SAMPLED ===
    'noise_std': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.3
    },
    'num_layers': {
        'distribution': 'uniform_int',
        'min': 2,
        'max': 5
    },
    'prior_mlp_activations': 'random',

    # === FIXED MLP ===
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 10,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False,

    # ==========================================================================
    # flexible_categorical hyperparameters
    # ==========================================================================

    # Classification: binary with balanced classes
    'num_classes': 2,
    'balanced': True,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.5,

    # Missing values: 10% random missing
    'nan_prob_no_reason': 0.1,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 1.0,

    # Features
    'num_features_used': 10,
    'categorical_feature_p': 0.0,
    'normalize_to_ranking': False,
    'normalize_by_used_features': False,
    'normalize_labels': True,
    'check_is_compatible': False,
}


if __name__ == "__main__":
    print("=" * 70)
    print("differentiable_prior + flexible_categorical Example")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use partial to bind mlp.get_batch to flexible_categorical
    # This creates: flexible_categorical.get_batch(..., get_batch=mlp.get_batch)
    fc_with_mlp = partial(flexible_categorical.get_batch, get_batch=mlp.get_batch)

    # Chain: diff_get_batch → flexible_categorical.get_batch → mlp.get_batch
    x, y, _ = diff_get_batch(
        batch_size=16,
        seq_len=100,
        num_features=10,
        get_batch=fc_with_mlp,
        device=device,
        hyperparameters=hyperparameters,
        batch_size_per_gp_sample=4,
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"Classes: {torch.unique(y[y != -100]).tolist()}")
    print(f"Missing: {torch.isnan(x).sum().item()}/{x.numel()} ({100*torch.isnan(x).float().mean():.1f}%)")
    print(f"\nPipeline: differentiable_prior → flexible_categorical → mlp")
    print(f"  - Sampled: noise_std, num_layers, activation")
    print(f"  - Post-processed: binary classification, missing values")
