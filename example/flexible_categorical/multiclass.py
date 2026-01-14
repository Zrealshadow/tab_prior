"""
Example 2: Multi-class Classification with Categorical Features
"""
import torch
from prior import flexible_categorical, mlp


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
    'prior_mlp_activations': 'relu',
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


if __name__ == "__main__":
    print("=" * 70)
    print("Example 2: 4-Class Classification with Categorical Features")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y, _ = flexible_categorical.get_batch(
        batch_size=8,
        seq_len=100,
        num_features=20,
        get_batch=mlp.get_batch,
        device=device,
        hyperparameters=config_multiclass,
        single_eval_pos=80
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"Classes: {torch.unique(y).tolist()}")
    print(f"Missing: {torch.isnan(x).sum().item()}/{x.numel()} ({100*torch.isnan(x).float().mean():.1f}%)")
