"""
MLP Prior: Multi-layer perceptron based synthetic data generation.

Generates tabular data using random MLPs with optional causal structure.
"""
import random
import math

import torch
from torch import nn
import numpy as np

from .utils import default_device, get_batch_to_dataloader


# =============================================================================
# Activation Functions
# =============================================================================

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'softplus': nn.Softplus,
}


def get_activation(name):
    """Get activation class from string name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: '{name}'. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]


# =============================================================================
# Helper Modules
# =============================================================================

class GaussianNoise(nn.Module):
    """Add Gaussian noise to inputs."""

    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


# =============================================================================
# Validation
# =============================================================================

def _validate_hyperparameters(hyperparameters):
    """Validate MLP hyperparameters."""
    # Required parameters
    required = [
        'num_layers', 'prior_mlp_hidden_dim', 'is_causal', 'num_causes',
        'noise_std', 'y_is_effect', 'pre_sample_weights', 'prior_mlp_dropout_prob',
        'pre_sample_causes', 'prior_mlp_activations', 'block_wise_dropout',
        'prior_mlp_scale_weights_sqrt', 'init_std', 'sampling'
    ]
    missing = [k for k in required if k not in hyperparameters]
    if missing:
        raise ValueError(f"Missing required MLP hyperparameters: {missing}")

    h = hyperparameters

    # Validate num_layers
    if h['num_layers'] < 2:
        raise ValueError(f"num_layers must be >= 2, got {h['num_layers']}")

    # Validate positive values
    if h['prior_mlp_hidden_dim'] <= 0:
        raise ValueError(f"prior_mlp_hidden_dim must be > 0, got {h['prior_mlp_hidden_dim']}")

    if h['num_causes'] <= 0:
        raise ValueError(f"num_causes must be > 0, got {h['num_causes']}")

    # Validate probabilities
    if not (0 <= h['prior_mlp_dropout_prob'] <= 1):
        raise ValueError(f"prior_mlp_dropout_prob must be in [0, 1], got {h['prior_mlp_dropout_prob']}")

    # Validate activation
    activation = h['prior_mlp_activations']
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: '{activation}'. Available: {list(ACTIVATIONS.keys())}")

    # Validate sampling
    if h['sampling'] not in ['normal', 'uniform', 'mixed']:
        raise ValueError(f"sampling must be 'normal', 'uniform', or 'mixed', got '{h['sampling']}'")


# =============================================================================
# MLP Model
# =============================================================================

def _sample_causes(num_causes):
    """Sample cause distribution parameters."""
    means = np.random.normal(0, 1, num_causes)
    std = np.abs(np.random.normal(0, 1, num_causes) * means)
    return means, std


class MLPGenerator(nn.Module):
    """MLP-based data generator."""

    def __init__(self, hyperparameters, seq_len, num_features, num_outputs, device, activation_class):
        super().__init__()

        # Store hyperparameters as attributes
        for key, value in hyperparameters.items():
            setattr(self, key, value)

        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.device = device
        self.activation_class = activation_class

        with torch.no_grad():
            self._build_network()

    def _build_network(self):
        """Build the MLP network."""
        # Adjust hidden dim for causal models
        if self.is_causal:
            self.prior_mlp_hidden_dim = max(
                self.prior_mlp_hidden_dim,
                self.num_outputs + 2 * self.num_features
            )
        else:
            self.num_causes = self.num_features

        # Pre-sample cause distributions
        if self.pre_sample_causes:
            means, std = _sample_causes(self.num_causes)
            self.causes_mean = torch.tensor(means, device=self.device).float()
            self.causes_mean = self.causes_mean.unsqueeze(0).unsqueeze(0).tile((self.seq_len, 1, 1))
            self.causes_std = torch.tensor(std, device=self.device).float()
            self.causes_std = self.causes_std.unsqueeze(0).unsqueeze(0).tile((self.seq_len, 1, 1))

        # Build layers
        self.layers = nn.Sequential(*self._create_layers())

        # Initialize weights
        self._init_weights()

        if getattr(self, 'verbose', False):
            print({
                'is_causal': self.is_causal,
                'num_causes': self.num_causes,
                'prior_mlp_hidden_dim': self.prior_mlp_hidden_dim,
                'num_layers': self.num_layers,
                'noise_std': self.noise_std,
                'y_is_effect': self.y_is_effect,
                'pre_sample_weights': self.pre_sample_weights,
                'prior_mlp_dropout_prob': self.prior_mlp_dropout_prob,
                'pre_sample_causes': self.pre_sample_causes
            })

    def _create_layers(self):
        """Create network layers."""
        layers = [nn.Linear(self.num_causes, self.prior_mlp_hidden_dim, device=self.device)]

        for _ in range(self.num_layers - 1):
            layers.extend(self._create_hidden_block(self.prior_mlp_hidden_dim))

        if not self.is_causal:
            layers.extend(self._create_hidden_block(self.num_outputs))

        return layers

    def _create_hidden_block(self, out_dim):
        """Create a hidden layer block with activation and noise."""
        if self.pre_sample_weights:
            noise_std = torch.abs(torch.normal(
                torch.zeros(size=(1, out_dim), device=self.device),
                float(self.noise_std)
            ))
        else:
            noise_std = float(self.noise_std)

        noise = GaussianNoise(noise_std, device=self.device)

        return [nn.Sequential(
            self.activation_class(),
            nn.Linear(self.prior_mlp_hidden_dim, out_dim),
            noise
        )]

    def _init_weights(self):
        """Initialize network weights with dropout."""
        for i, (name, param) in enumerate(self.layers.named_parameters()):
            if len(param.shape) != 2:  # Skip biases
                continue

            if self.block_wise_dropout:
                self._init_blockwise(param)
            else:
                self._init_standard(param, i)

    def _init_blockwise(self, param):
        """Block-wise dropout initialization."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape[0], param.shape[1]))))
        w, h = param.shape[0] // n_blocks, param.shape[1] // n_blocks
        keep_prob = (n_blocks * w * h) / param.numel()

        scale = 1 / (keep_prob ** (0.5 if self.prior_mlp_scale_weights_sqrt else 1))

        for block in range(n_blocks):
            nn.init.normal_(
                param[w * block:w * (block + 1), h * block:h * (block + 1)],
                std=self.init_std * scale
            )

    def _init_standard(self, param, layer_idx):
        """Standard dropout initialization."""
        dropout_prob = self.prior_mlp_dropout_prob if layer_idx > 0 else 0.0
        dropout_prob = min(dropout_prob, 0.99)

        scale = 1 / (1 - dropout_prob ** (0.5 if self.prior_mlp_scale_weights_sqrt else 1))
        nn.init.normal_(param, std=self.init_std * scale)

        # Apply dropout mask
        mask = torch.bernoulli(torch.zeros_like(param) + 1 - dropout_prob)
        param *= mask

    def forward(self):
        """Generate synthetic data."""
        # Sample causes
        causes = self._sample_causes()

        # Forward through network
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]  # Skip input and first hidden

        # Extract x and y
        if self.is_causal:
            x, y = self._extract_causal(outputs)
        else:
            x = causes
            y = outputs[-1]

        # Check for NaN
        if torch.isnan(x).any() or torch.isnan(y).any():
            print(f'NaN detected in MLP: x={torch.isnan(x).sum()}, y={torch.isnan(y).sum()}')
            x.fill_(0.0)
            y.fill_(-100)  # ignore index for CE loss

        # Random feature rotation
        if getattr(self, 'random_feature_rotation', False):
            shift = random.randrange(x.shape[-1])
            idx = (torch.arange(x.shape[-1], device=self.device) + shift) % x.shape[-1]
            x = x[..., idx]

        return x, y

    def _sample_causes(self):
        """Sample input causes based on sampling mode."""
        if self.sampling == 'normal':
            return self._sample_normal()
        elif self.sampling == 'uniform':
            return torch.rand((self.seq_len, 1, self.num_causes), device=self.device)
        elif self.sampling == 'mixed':
            return self._sample_mixed()
        else:
            raise ValueError(f"Invalid sampling mode: {self.sampling}")

    def _sample_normal(self):
        """Sample from normal distribution."""
        if self.pre_sample_causes:
            return torch.normal(self.causes_mean, self.causes_std.abs()).float()
        return torch.normal(0., 1., (self.seq_len, 1, self.num_causes), device=self.device).float()

    def _sample_mixed(self):
        """Sample from mixture of distributions."""
        zipf_p = random.random() * 0.66
        multi_p = random.random() * 0.66
        normal_p = random.random() * 0.66

        causes = []
        for n in range(self.num_causes):
            if random.random() > normal_p:
                # Normal
                if self.pre_sample_causes:
                    c = torch.normal(self.causes_mean[:, :, n], self.causes_std[:, :, n].abs()).float()
                else:
                    c = torch.normal(0., 1., (self.seq_len, 1), device=self.device).float()
            elif random.random() > multi_p:
                # Multinomial
                c = torch.multinomial(
                    torch.rand(random.randint(2, 10)),
                    self.seq_len, replacement=True
                ).to(self.device).unsqueeze(-1).float()
                c = (c - c.mean()) / c.std()
            else:
                # Zipf
                c = torch.tensor(
                    np.random.zipf(2.0 + random.random() * 2, size=self.seq_len),
                    device=self.device
                ).unsqueeze(-1).float()
                c = torch.minimum(c, torch.tensor(10.0, device=self.device))
                c = c - c.mean()
            causes.append(c.unsqueeze(-1))

        return torch.cat(causes, dim=-1)

    def _extract_causal(self, outputs):
        """Extract x and y from causal graph."""
        outputs_flat = torch.cat(outputs, dim=-1)

        if getattr(self, 'in_clique', False):
            start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
            perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
        else:
            perm = torch.randperm(outputs_flat.shape[-1] - 1, device=self.device)

        # Select y indices
        if self.y_is_effect:
            y_idx = list(range(-self.num_outputs, 0))
        else:
            y_idx = perm[:self.num_outputs]

        # Select x indices
        x_idx = perm[self.num_outputs:self.num_outputs + self.num_features]
        if getattr(self, 'sort_features', False):
            x_idx, _ = torch.sort(x_idx)

        y = outputs_flat[:, :, y_idx]
        x = outputs_flat[:, :, x_idx]

        return x, y


# =============================================================================
# Batch Generation
# =============================================================================

def get_batch(batch_size, seq_len, num_features, hyperparameters, device=default_device,
              num_outputs=1, sampling='normal', **kwargs):
    """
    Generate a batch of synthetic data using random MLPs.

    Args:
        batch_size: Number of samples in batch
        seq_len: Sequence length (number of data points)
        num_features: Number of features
        hyperparameters: Dict containing:
            - num_layers: Number of MLP layers (>= 2)
            - prior_mlp_hidden_dim: Hidden layer dimension
            - is_causal: Use causal structure
            - num_causes: Number of causal variables
            - noise_std: Gaussian noise standard deviation
            - y_is_effect: Target is effect (not cause)
            - pre_sample_weights: Pre-sample weight distributions
            - prior_mlp_dropout_prob: Dropout probability
            - pre_sample_causes: Pre-sample cause distributions
            - prior_mlp_activations: Activation function name
            - block_wise_dropout: Use block-wise dropout
            - prior_mlp_scale_weights_sqrt: Scale weights by sqrt
            - init_std: Weight initialization std
            - sampling: Input sampling mode ('normal', 'uniform', 'mixed')
        device: torch device
        num_outputs: Number of output dimensions

    Returns:
        x: Features tensor (seq_len, batch_size, num_features)
        y: Targets tensor (seq_len, batch_size)
        y_: Copy of targets
    """
    _validate_hyperparameters(hyperparameters)

    # Handle multi-node classification
    if hyperparameters.get('multiclass_type') == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    # Get activation class
    activation_class = get_activation(hyperparameters['prior_mlp_activations'])

    # Create model(s)
    if hyperparameters.get('new_mlp_per_example', False):
        def get_model():
            return MLPGenerator(
                hyperparameters, seq_len, num_features, num_outputs, device, activation_class
            ).to(device)
    else:
        model = MLPGenerator(
            hyperparameters, seq_len, num_features, num_outputs, device, activation_class
        ).to(device)
        get_model = lambda: model

    # Generate samples
    samples = [get_model()() for _ in range(batch_size)]

    # Concatenate results
    x, y = zip(*samples)
    x = torch.cat(x, dim=1).detach()
    y = torch.cat(y, dim=1).detach().squeeze(2)

    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)
