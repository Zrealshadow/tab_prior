"""
Example 1: Binary Classification with Missing Data
"""
import torch
from prior import flexible_categorical, mlp


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


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Binary Classification with Missing Data")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y, _ = flexible_categorical.get_batch(
        batch_size=8,
        seq_len=100,
        num_features=20,
        get_batch=mlp.get_batch,
        device=device,
        hyperparameters=config_binary
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"Classes: {torch.unique(y).tolist()}")
    print(f"Missing: {torch.isnan(x).sum().item()}/{x.numel()} ({100*torch.isnan(x).float().mean():.1f}%)")
