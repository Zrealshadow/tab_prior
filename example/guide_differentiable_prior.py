"""
Complete Guide: How to Use differentiable_prior
For meta-learning and automatic hyperparameter variation
"""
import torch
from prior.differentiable_prior import get_batch as diff_get_batch
from prior import mlp, fast_gp

print("=" * 70)
print("differentiable_prior Usage Guide")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# EXAMPLE 1: Basic Usage - Vary noise_std
# ============================================================================

print("\n" + "=" * 70)
print("Example 1: Varying a Single Hyperparameter (noise_std)")
print("=" * 70)

# Define WHICH hyperparameters to vary and HOW
differentiable_hyperparameters_1 = {
    'noise_std': {
        'distribution': 'meta_trunc_norm_log_scaled',  # Distribution type
        'min_mean': 0.01,      # Minimum value
        'max_mean': 0.5,       # Maximum value
        'round': False,        # Don't round to integer
        'lower_bound': 0.0     # Hard lower bound
    }
}

# Define FIXED hyperparameters (same for all batches)
fixed_hyperparameters = {
    # MLP parameters (all required)
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': True,
    'num_causes': 10,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
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

    # REQUIRED for differentiable_prior
    'emsize': 64  # Embedding dimension for hyperparameter indicators
}

# Generate data with sampled hyperparameters
x, y, y_clean, hp_embeddings = diff_get_batch(
    batch_size=8,
    seq_len=100,
    num_features=10,
    get_batch=mlp.get_batch,  # Use MLP as base method
    device=device,
    differentiable_hyperparameters=differentiable_hyperparameters_1,
    hyperparameters=fixed_hyperparameters
)

print(f"\nGenerated data:")
print(f"  x shape: {x.shape}")
print(f"  y shape: {y.shape}")
print(f"  hp_embeddings shape: {hp_embeddings.shape if hp_embeddings is not None else None}")
print(f"\nHyperparameter embeddings are normalized indicators of sampled values")
print(f"Each batch has different noise_std sampled from [0.01, 0.5]")


# ============================================================================
# EXAMPLE 2: Vary Multiple Hyperparameters
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: Varying Multiple Hyperparameters")
print("=" * 70)

differentiable_hyperparameters_2 = {
    # Vary num_layers (integer)
    'num_layers': {
        'distribution': 'meta_gamma',  # Good for integers
        'max_alpha': 3,
        'max_scale': 2,
        'round': True,        # Round to integer
        'lower_bound': 2      # Minimum 2 layers
    },

    # Vary noise_std (continuous)
    'noise_std': {
        'distribution': 'meta_trunc_norm_log_scaled',
        'min_mean': 0.01,
        'max_mean': 0.3,
        'round': False,
        'lower_bound': 0.0
    },

    # Vary is_causal (categorical choice)
    'is_causal': {
        'distribution': 'meta_choice',
        'choice_values': [True, False]  # Choose between True/False
    }
}

fixed_hyperparameters_2 = {
    'prior_mlp_hidden_dim': 64,
    'num_causes': 10,
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
    'emsize': 64
}

x, y, y_clean, hp_embeddings = diff_get_batch(
    batch_size=16,
    seq_len=100,
    num_features=15,
    get_batch=mlp.get_batch,
    device=device,
    differentiable_hyperparameters=differentiable_hyperparameters_2,
    hyperparameters=fixed_hyperparameters_2,
    batch_size_per_gp_sample=4  # Process in sub-batches of 4
)

print(f"\nGenerated data with 3 varied hyperparameters:")
print(f"  x shape: {x.shape}")
print(f"  y shape: {y.shape}")
print(f"  hp_embeddings shape: {hp_embeddings.shape}")
print(f"  → Each row in hp_embeddings describes the hyperparameters for that batch")
print(f"  → Embeddings have {hp_embeddings.shape[1]} dimensions (one per varied hyperparameter)")


# ============================================================================
# EXAMPLE 3: Vary Activation Functions
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Varying Activation Functions")
print("=" * 70)

differentiable_hyperparameters_3 = {
    'prior_mlp_activations': {
        'distribution': 'meta_choice_mixed',  # For callables
        'choice_values': [
            lambda: torch.nn.ReLU(),
            lambda: torch.nn.Tanh(),
            lambda: torch.nn.LeakyReLU()
        ]
    },
    'noise_std': {
        'distribution': 'uniform',  # Simple uniform
        'min': 0.05,
        'max': 0.2
    }
}

fixed_hyperparameters_3 = {
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
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
    'emsize': 64
}

x, y, y_clean, hp_embeddings = diff_get_batch(
    batch_size=8,
    seq_len=100,
    num_features=10,
    get_batch=mlp.get_batch,
    device=device,
    differentiable_hyperparameters=differentiable_hyperparameters_3,
    hyperparameters=fixed_hyperparameters_3
)

print(f"\nGenerated with varied activations:")
print(f"  x shape: {x.shape}")
print(f"  hp_embeddings shape: {hp_embeddings.shape}")
print(f"  Each batch used a different activation function")


# ============================================================================
# EXAMPLE 4: Use with GP instead of MLP
# ============================================================================

print("\n" + "=" * 70)
print("Example 4: Using differentiable_prior with GP")
print("=" * 70)

differentiable_hyperparameters_gp = {
    'lengthscale': {
        'distribution': 'meta_trunc_norm_log_scaled',
        'min_mean': 0.1,
        'max_mean': 2.0,
        'round': False,
        'lower_bound': 0.05
    },
    'noise': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.3
    }
}

fixed_hyperparameters_gp = {
    'outputscale': 1.0,
    'sampling': 'uniform',
    'fast_computations': (True, True, True),
    'emsize': 64
}

x, y, y_clean, hp_embeddings = diff_get_batch(
    batch_size=8,
    seq_len=100,
    num_features=10,
    get_batch=fast_gp.get_batch,  # Use GP instead of MLP
    device=device,
    differentiable_hyperparameters=differentiable_hyperparameters_gp,
    hyperparameters=fixed_hyperparameters_gp
)

print(f"\nGP with varied lengthscale and noise:")
print(f"  x shape: {x.shape}")
print(f"  hp_embeddings shape: {hp_embeddings.shape}")


# ============================================================================
# EXAMPLE 5: Training Loop Pattern (Meta-Learning)
# ============================================================================

print("\n" + "=" * 70)
print("Example 5: Typical Meta-Learning Training Pattern")
print("=" * 70)

print("""
# Pseudo-code for training a meta-learning model (like TabPFN):

for epoch in range(num_epochs):
    for batch_idx in range(batches_per_epoch):
        # Generate data with random hyperparameters
        x, y, _, hp_embeddings = diff_get_batch(
            batch_size=32,
            seq_len=100,
            num_features=20,
            get_batch=mlp.get_batch,
            device=device,
            differentiable_hyperparameters=diff_hparams,
            hyperparameters=fixed_hparams
        )

        # Forward pass: model receives both data AND hyperparameter info
        predictions = model(x, y, hp_embeddings)

        # The model learns to adapt based on hp_embeddings
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

# The model learns: "When noise_std is high (from embedding),
# I should be more conservative in predictions"
""")


# ============================================================================
# Distribution Types Summary
# ============================================================================

print("\n" + "=" * 70)
print("Available Distribution Types")
print("=" * 70)

summary = """
┌─────────────────────────────────────────────────────────────────┐
│ Distribution Types                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. uniform                                                      │
│    - Simple uniform distribution                                │
│    - Parameters: min, max                                       │
│    - Use for: Any continuous value in range                     │
│                                                                 │
│ 2. uniform_int                                                  │
│    - Integer uniform distribution                               │
│    - Parameters: min, max                                       │
│    - Use for: Integer hyperparameters                          │
│                                                                 │
│ 3. meta_gamma                                                   │
│    - Gamma distribution (good for positive integers)            │
│    - Parameters: max_alpha, max_scale, round, lower_bound      │
│    - Use for: num_layers, num_causes, etc.                     │
│                                                                 │
│ 4. meta_trunc_norm_log_scaled                                   │
│    - Log-scaled truncated normal                                │
│    - Parameters: min_mean, max_mean, lower_bound, round        │
│    - Use for: noise_std, learning_rate, dropout_prob           │
│                                                                 │
│ 5. meta_choice                                                  │
│    - Categorical choice (for discrete values)                   │
│    - Parameters: choice_values (list)                          │
│    - Use for: is_causal (True/False), sampling method          │
│                                                                 │
│ 6. meta_choice_mixed                                            │
│    - Categorical choice for callables/functions                 │
│    - Parameters: choice_values (list of lambdas)               │
│    - Use for: activation functions, samplers                    │
└─────────────────────────────────────────────────────────────────┘

Key Points:
- hp_embeddings are NORMALIZED indicators (roughly in [-1, 1] range)
- They describe WHAT hyperparameters were used for each batch
- Your model can use these to adapt its behavior
- Essential for meta-learning (TabPFN-style training)
"""

print(summary)

print("\n✅ differentiable_prior guide completed!\n")
