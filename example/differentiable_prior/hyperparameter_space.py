"""
Example: Using HyperparameterSpace directly

HyperparameterSpace separates sampling from data generation,
useful when you want to:
- Inspect sampled values before generating data
- Use the same sampled config multiple times
- Integrate with custom training loops
"""
import torch
from prior.differentiable_prior import HyperparameterSpace
from prior import mlp


# Define the hyperparameter space
hp_space = HyperparameterSpace({
    # === SAMPLED ===
    'noise_std': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
    'num_layers': {'distribution': 'uniform_int', 'min': 2, 'max': 5},
    'is_causal': {'distribution': 'meta_choice', 'choice_values': [True, False]},

    # === FIXED ===
    'prior_mlp_hidden_dim': 64,
    'num_causes': 10,
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
})


if __name__ == "__main__":
    print("=" * 70)
    print("HyperparameterSpace Example")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sample and inspect configs
    print("\nSampling 5 configs:")
    for i in range(5):
        config = hp_space.sample()
        print(f"  {i+1}: noise_std={config['noise_std']:.3f}, "
              f"num_layers={config['num_layers']}, "
              f"is_causal={config['is_causal']}")

    # Use a sampled config to generate data
    print("\nGenerating data with a sampled config:")
    config = hp_space.sample()
    print(f"  Config: noise_std={config['noise_std']:.3f}, "
          f"num_layers={config['num_layers']}, is_causal={config['is_causal']}")

    x, y, _ = mlp.get_batch(
        batch_size=8,
        seq_len=100,
        num_features=10,
        device=device,
        hyperparameters=config
    )

    print(f"  Output: x{x.shape}, y{y.shape}")
