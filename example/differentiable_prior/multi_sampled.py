"""
Example: Multiple sampled hyperparameters

Demonstrates different distribution types for sampling hyperparameters.
"""
import torch
from prior.differentiable_prior import get_batch
from prior import mlp


hyperparameters = {
    # === SAMPLED ===

    # meta_gamma: good for positive integers (num_layers, num_causes)
    'num_layers': {
        'distribution': 'meta_gamma',
        'max_alpha': 3,
        'max_scale': 2,
        'round': True,
        'lower_bound': 2
    },

    # meta_trunc_norm_log_scaled: good for values spanning orders of magnitude
    'noise_std': {
        'distribution': 'meta_trunc_norm_log_scaled',
        'min_mean': 0.01,
        'max_mean': 0.3,
    },

    # meta_choice: good for categorical choices
    'is_causal': {
        'distribution': 'meta_choice',
        'choice_values': [True, False]
    },

    # meta_choice: also works for activation functions
    'prior_mlp_activations': 'random',

    # === FIXED ===
    'prior_mlp_hidden_dim': 64,
    'num_causes': 10,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.15,
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
}


if __name__ == "__main__":
    print("=" * 70)
    print("Multiple Sampled Hyperparameters Example")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # batch_size=16, batch_size_per_gp_sample=4 → 4 sub-batches with different configs
    x, y, y_clean = get_batch(
        batch_size=16,
        seq_len=100,
        num_features=15,
        get_batch=mlp.get_batch,
        device=device,
        hyperparameters=hyperparameters,
        batch_size_per_gp_sample=4
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"4 sub-batches, each with different:")
    print(f"  - num_layers (gamma, >= 2)")
    print(f"  - noise_std (log-scaled, [0.01, 0.3])")
    print(f"  - is_causal (True/False)")
    print(f"  - prior_mlp_activations (relu/tanh/leaky_relu)")
