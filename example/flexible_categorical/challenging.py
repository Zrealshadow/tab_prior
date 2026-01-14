"""
Example 4: High Missing Rate + Categorical (Challenging Dataset)
"""
import torch
from prior import flexible_categorical, mlp


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
    'prior_mlp_activations': 'tanh',
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
    'nan_prob_no_reason': 0.9,
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


if __name__ == "__main__":
    print("=" * 70)
    print("Example 4: Challenging Dataset (High Missing + Categorical)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y, _ = flexible_categorical.get_batch(
        batch_size=8,
        seq_len=100,
        num_features=20,
        get_batch=mlp.get_batch,
        device=device,
        hyperparameters=config_challenging,
        single_eval_pos=80
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"Classes: {torch.unique(y).tolist()}")
    print(f"Missing: {torch.isnan(x).sum().item()}/{x.numel()} ({100*torch.isnan(x).float().mean():.1f}%)")
