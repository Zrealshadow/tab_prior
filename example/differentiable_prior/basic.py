"""
Example: Basic differentiable_prior usage

Demonstrates mixing fixed and sampled hyperparameters in a single dict.
The module auto-detects which values to sample based on 'distribution' key.
"""
import torch
from prior.differentiable_prior import get_batch
from prior import mlp


# All hyperparameters in one dict:
# - Plain values → fixed
# - Dict with 'distribution' → sampled per sub-batch
hyperparameters = {
    # === SAMPLED ===
    'noise_std': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.5
    },
    'num_layers': {
        'distribution': 'uniform_int',
        'min': 2,
        'max': 7
    },

    'prior_mlp_hidden_dim': {
        'distribution': 'meta_choice',
        'choice_values': [32, 64, 128, 256]
    },

    'sampling': {
        'distribution': 'meta_choice',
        'choice_values': ['normal', 'uniform', 'mixed']
    },



    # === FIXED ===
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
    'new_mlp_per_example': False,
    'prior_mlp_activations': 'random',
}


if __name__ == "__main__":
    print("=" * 70)
    print("Basic differentiable_prior Example")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y, y_clean = get_batch(
        batch_size=256,
        seq_len=100,
        num_features=10,
        get_batch=mlp.get_batch,
        device=device,
        hyperparameters=hyperparameters
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
