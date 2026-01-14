"""
Prior Bag: Mixture of different data generation methods.

Randomly samples from multiple generators (e.g., GP, MLP) based on weights.
"""
import torch

from .utils import get_batch_to_dataloader, default_device


def _validate_hyperparameters(hyperparameters):
    """Validate prior_bag hyperparameters."""
    # Check required parameters
    if 'prior_bag_get_batch' not in hyperparameters:
        raise ValueError("Missing required hyperparameter: 'prior_bag_get_batch' (list of get_batch functions)")

    priors = hyperparameters['prior_bag_get_batch']

    if not isinstance(priors, (list, tuple)):
        raise ValueError(f"'prior_bag_get_batch' must be a list, got {type(priors)}")

    if len(priors) < 1:
        raise ValueError("'prior_bag_get_batch' must contain at least one generator")

    # Check weights for additional priors
    for i in range(1, len(priors)):
        weight_key = f'prior_bag_exp_weights_{i}'
        if weight_key not in hyperparameters:
            raise ValueError(f"Missing weight for prior {i}: '{weight_key}'")

        weight = hyperparameters[weight_key]
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"'{weight_key}' must be a non-negative number, got {weight}")


def get_batch(batch_size, seq_len, num_features, device=default_device,
              hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    """
    Generate a batch by sampling from multiple generators.

    Args:
        batch_size: Total batch size
        seq_len: Sequence length (number of samples)
        num_features: Number of features
        device: torch device
        hyperparameters: Dict containing:
            - prior_bag_get_batch: List of get_batch functions [gp, mlp, ...]
            - prior_bag_exp_weights_1: Weight for 2nd generator (1st always has weight 1.0)
            - prior_bag_exp_weights_2: Weight for 3rd generator
            - etc.
        batch_size_per_gp_sample: Sub-batch size for each generator call

    Returns:
        x: Features tensor (seq_len, batch_size, num_features)
        y: Targets tensor (seq_len, batch_size)
        y_: Copy of targets
    """
    _validate_hyperparameters(hyperparameters)

    # Setup batching
    batch_size_per_gp_sample = batch_size_per_gp_sample or min(64, batch_size)
    num_models = batch_size // batch_size_per_gp_sample

    if num_models == 0:
        raise ValueError(f"batch_size ({batch_size}) too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})")
    if num_models * batch_size_per_gp_sample != batch_size:
        raise ValueError(f"batch_size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})")

    # Get generators and their weights
    generators = hyperparameters['prior_bag_get_batch']
    weights = [1.0] + [
        hyperparameters[f'prior_bag_exp_weights_{i}']
        for i in range(1, len(generators))
    ]

    # Sample which generator to use for each sub-batch
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    probs = torch.softmax(weights_tensor, dim=0)
    assignments = torch.multinomial(probs, num_models, replacement=True).numpy()

    if hyperparameters.get('verbose', False):
        print(f'PRIOR_BAG: weights={weights_tensor.tolist()}, assignments={assignments}')

    # Generate samples from assigned generators
    args = {
        'device': device,
        'seq_len': seq_len,
        'num_features': num_features,
        'batch_size': batch_size_per_gp_sample,
        **kwargs
    }

    samples = [
        generators[int(idx)](hyperparameters=hyperparameters, **args)
        for idx in assignments
    ]

    # Concatenate results
    x, y, y_ = zip(*samples)
    x = torch.cat(x, dim=1).detach()
    y = torch.cat(y, dim=1).detach()
    y_ = torch.cat(y_, dim=1).detach()

    return x, y, y_


DataLoader = get_batch_to_dataloader(get_batch)
