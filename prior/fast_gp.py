"""
Fast GP: Gaussian Process based data generation.

Uses GPyTorch for efficient GP sampling.
"""
import torch
from torch import nn
import gpytorch

from .utils import get_batch_to_dataloader, default_device


# =============================================================================
# GP Model
# =============================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    """Simple GP model with RBF kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _create_model(x, y, hyperparameters):
    """Create and configure GP model with given hyperparameters."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-9)
    )
    model = ExactGPModel(x, y, likelihood)

    # Set kernel hyperparameters
    model.likelihood.noise = torch.ones_like(model.likelihood.noise) * hyperparameters['noise']
    model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale) * hyperparameters['outputscale']
    model.covar_module.base_kernel.lengthscale = (
        torch.ones_like(model.covar_module.base_kernel.lengthscale) * hyperparameters['lengthscale']
    )

    return model, likelihood


# =============================================================================
# Validation
# =============================================================================

def _validate_hyperparameters(hyperparameters):
    """Validate GP hyperparameters."""
    required = ['noise', 'outputscale', 'lengthscale']
    missing = [k for k in required if k not in hyperparameters]
    if missing:
        raise ValueError(f"Missing required GP hyperparameters: {missing}")

    # Validate positive values
    for param in required:
        val = hyperparameters[param]
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"'{param}' must be a positive number, got {val}")

    # Validate sampling mode
    sampling = hyperparameters.get('sampling', 'uniform')
    if sampling not in ['uniform', 'normal']:
        raise ValueError(f"'sampling' must be 'uniform' or 'normal', got '{sampling}'")


def _normalize_hyperparameters(hyperparameters):
    """Convert legacy tuple/list format to dict."""
    if isinstance(hyperparameters, (tuple, list)):
        return {
            'noise': hyperparameters[0],
            'outputscale': hyperparameters[1],
            'lengthscale': hyperparameters[2],
            'is_binary_classification': hyperparameters[3] if len(hyperparameters) > 3 else False,
            'normalize_by_used_features': hyperparameters[5] if len(hyperparameters) > 5 else False,
            'order_y': hyperparameters[6] if len(hyperparameters) > 6 else False,
            'sampling': hyperparameters[7] if len(hyperparameters) > 7 else 'uniform',
        }
    elif hyperparameters is None:
        return {'noise': 0.1, 'outputscale': 0.1, 'lengthscale': 0.1, 'sampling': 'uniform'}
    return hyperparameters


# =============================================================================
# Batch Generation
# =============================================================================

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              equidistant_x=False, fix_x=None, **kwargs):
    """
    Generate a batch of data using Gaussian Process.

    Args:
        batch_size: Number of independent GP samples
        seq_len: Sequence length (number of points per sample)
        num_features: Input dimensionality
        device: torch device
        hyperparameters: Dict containing:
            - noise: Observation noise variance
            - outputscale: Kernel output scale
            - lengthscale: Kernel length scale
            - sampling: 'uniform' or 'normal' for input distribution
        equidistant_x: Use equidistant x values (only for 1D)
        fix_x: Fixed input tensor to use

    Returns:
        x: Features tensor (seq_len, batch_size, num_features)
        y: Targets tensor (seq_len, batch_size)
        y_: Copy of targets
    """
    # Normalize and validate hyperparameters
    hyperparameters = _normalize_hyperparameters(hyperparameters)
    _validate_hyperparameters(hyperparameters)

    if hyperparameters.get('verbose', False):
        print({
            'noise': hyperparameters['noise'],
            'outputscale': hyperparameters['outputscale'],
            'lengthscale': hyperparameters['lengthscale'],
            'batch_size': batch_size,
            'sampling': hyperparameters.get('sampling', 'uniform')
        })

    # Validate x options
    if equidistant_x and fix_x is not None:
        raise ValueError("Cannot use both equidistant_x and fix_x")

    # Generate input features
    x = _generate_inputs(batch_size, seq_len, num_features, device, hyperparameters, equidistant_x, fix_x)

    # Sample from GP
    sample = _sample_gp(x, hyperparameters, device)

    return x.transpose(0, 1), sample, sample


def _generate_inputs(batch_size, seq_len, num_features, device, hyperparameters, equidistant_x, fix_x):
    """Generate input features x."""
    if equidistant_x:
        if num_features != 1:
            raise ValueError(f"equidistant_x requires num_features=1, got {num_features}")
        return torch.linspace(0, 1., seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1).to(device)

    if fix_x is not None:
        if fix_x.shape != (seq_len, num_features):
            raise ValueError(f"fix_x shape must be ({seq_len}, {num_features}), got {fix_x.shape}")
        return fix_x.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    # Random inputs
    sampling = hyperparameters.get('sampling', 'uniform')
    if sampling == 'uniform':
        return torch.rand(batch_size, seq_len, num_features, device=device)
    else:
        return torch.randn(batch_size, seq_len, num_features, device=device)


def _sample_gp(x, hyperparameters, device):
    """Sample from GP prior."""
    fast_computations = hyperparameters.get('fast_computations', (True, True, True))

    with gpytorch.settings.fast_computations(*fast_computations):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with gpytorch.settings.prior_mode(True):
                    model, likelihood = _create_model(x, torch.Tensor(), hyperparameters)
                    model.to(device)

                    dist = model(x)
                    dist = likelihood(dist)
                    sample = dist.sample().transpose(0, 1)
                    return sample

            except RuntimeError as e:
                # Can happen when torch.linalg.eigh fails
                if attempt < max_retries - 1:
                    print(f'GP sampling failed (attempt {attempt + 1}), retrying...')
                else:
                    raise RuntimeError(f"GP sampling failed after {max_retries} attempts: {e}")


# =============================================================================
# Evaluation
# =============================================================================

def get_model_on_device(x, y, hyperparameters, device):
    """Create and move GP model to device."""
    model, likelihood = _create_model(x, y, hyperparameters)
    model.to(device)
    return model, likelihood


@torch.no_grad()
def evaluate(x, y, y_non_noisy, use_mse=False, hyperparameters=None,
             get_model_on_device=get_model_on_device, device=default_device,
             step_size=1, start_pos=0):
    """
    Evaluate GP predictions at each timestep.

    Args:
        x: Input features (seq_len, batch_size, num_features)
        y: Targets (seq_len, batch_size)
        y_non_noisy: Clean targets (unused, for API compatibility)
        use_mse: Use MSE loss instead of negative log-likelihood
        hyperparameters: GP hyperparameters
        device: torch device
        step_size: Evaluate every step_size timesteps
        start_pos: Starting position for evaluation

    Returns:
        all_losses: Loss at each timestep for each batch
        mean_losses: Mean loss at each timestep
        elapsed_time: Time taken for evaluation
    """
    import time
    start_time = time.time()

    hyperparameters = hyperparameters or {}
    losses_after_t = [0.0] if start_pos == 0 else []
    all_losses_after_t = []

    fast_computations = hyperparameters.get('fast_computations', (True, True, True))

    with gpytorch.settings.fast_computations(*fast_computations), gpytorch.settings.fast_pred_var(False):
        for t in range(max(start_pos, 1), len(x), step_size):
            # Fit GP on data up to time t
            model, likelihood = get_model_on_device(
                x[:t].transpose(0, 1),
                y[:t].transpose(0, 1),
                hyperparameters,
                device
            )
            model.eval()

            # Predict at time t
            f = model(x[t].unsqueeze(1))
            pred = likelihood(f)
            means = pred.mean.squeeze()

            # Compute loss
            if use_mse:
                loss = nn.MSELoss(reduction='none')(means, y[t])
            else:
                loss = -pred.log_prob(y[t].unsqueeze(1))

            losses_after_t.append(loss.mean())
            all_losses_after_t.append(loss.flatten())

    return (
        torch.stack(all_losses_after_t).cpu(),
        torch.tensor(losses_after_t).cpu(),
        time.time() - start_time
    )


# =============================================================================
# DataLoader
# =============================================================================

DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1


if __name__ == '__main__':
    hps = {'noise': 0.1, 'outputscale': 0.1, 'lengthscale': 0.1}
    print(evaluate(*get_batch(1000, 10, hyperparameters=hps, num_features=10), use_mse=False, hyperparameters=hps))
