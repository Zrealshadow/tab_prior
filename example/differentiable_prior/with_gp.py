"""
Example: Using differentiable_prior with GP prior

Demonstrates sampling GP-specific hyperparameters (lengthscale, noise).
"""
import torch
from prior.differentiable_prior import get_batch
from prior import fast_gp


hyperparameters = {
    # === SAMPLED ===
    'lengthscale': {
        'distribution': 'meta_trunc_norm_log_scaled',
        'min_mean': 0.1,
        'max_mean': 2.0,
        'lower_bound': 0.05
    },
    'noise': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.3
    },

    # === FIXED ===
    'outputscale': 1.0,
    'sampling': 'uniform',
    'fast_computations': (True, True, True),
}


if __name__ == "__main__":
    print("=" * 70)
    print("differentiable_prior with GP Example")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y, y_clean = get_batch(
        batch_size=8,
        seq_len=100,
        num_features=10,
        get_batch=fast_gp.get_batch,
        device=device,
        hyperparameters=hyperparameters
    )

    print(f"\nOutput: x{x.shape}, y{y.shape}")
    print(f"GP with sampled lengthscale and noise")
