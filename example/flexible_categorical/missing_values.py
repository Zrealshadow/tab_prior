"""
Example: Missing Values Demo

Demonstrates NaN injection with high probability settings.
Note: Due to random gates in _add_missing_values(), run multiple times
to see missing values (~50% of runs will have NaN).
"""
import torch
from prior import flexible_categorical, mlp


config_missing = {
    # Base MLP
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
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

    # Binary classification
    'num_classes': 2,
    'balanced': True,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,

    # High missing rate - set to 1.0 to pass inner random gate
    'nan_prob_no_reason': 0.2,
    'nan_prob_a_reason': 0,
    'nan_prob_unknown_reason': 0,
    'set_value_to_nan': 1.0,  # Always use NaN (not sentinel values)

    'categorical_feature_p': 0.0,
    'normalize_to_ranking': False,
    'normalize_by_used_features': False,
    'num_features_used': 10,
    'normalize_labels': True,
    'check_is_compatible': False,
}


if __name__ == "__main__":
    print("=" * 70)
    print("Example: Missing Values Demo")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run multiple times to demonstrate randomness
    for i in range(100):
        x, y, _ = flexible_categorical.get_batch(
            batch_size=2,
            seq_len=100,
            num_features=30,
            get_batch=mlp.get_batch,
            device=device,
            hyperparameters=config_missing
        )

        missing_count = torch.isnan(x).sum().item()
        missing_pct = 100 * torch.isnan(x).float().mean()
        print(f"Run {i+1}: Missing {missing_count}/{x.numel()} ({missing_pct:.1f}%)")
