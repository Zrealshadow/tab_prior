"""
Differentiable Prior: Random hyperparameter sampling for diverse data generation.

This module provides a configuration space for hyperparameters that can be sampled
to generate diverse synthetic tabular datasets. Each batch can have different
hyperparameters (e.g., different num_layers, noise levels, activations).

Main use case: Training meta-learning models (like TabPFN) on varied synthetic data.
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
from torch import nn

from .utils import get_batch_to_dataloader


# =============================================================================
# Sampler Interface (Abstract Base Class)
# =============================================================================

class Sampler(ABC):
    """Abstract base class for all hyperparameter samplers."""

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from the distribution."""
        pass

    def __call__(self) -> Any:
        """Allow calling sampler directly."""
        return self.sample()


# =============================================================================
# Global Registry: Maps distribution names to Config classes
# =============================================================================

_CONFIG_REGISTRY: dict[str, type] = {}


def register_config(name: str):
    """Decorator to register a config class with a distribution name."""
    def decorator(cls):
        _CONFIG_REGISTRY[name] = cls
        return cls
    return decorator


def get_registered_configs() -> dict[str, type]:
    """Get all registered config classes."""
    return _CONFIG_REGISTRY.copy()


# =============================================================================
# Sampler Config Classes with Validation
# =============================================================================

@dataclass
class SamplerConfig(ABC):
    """Base class for sampler configurations."""

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters. Raises ValueError if invalid."""
        pass

    @abstractmethod
    def create_sampler(self) -> Sampler:
        """Create a sampler instance from this config."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict) -> 'SamplerConfig':
        """
        Construct a config instance from a dictionary.

        Args:
            config: Dict with distribution-specific parameters

        Returns:
            SamplerConfig instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        pass

    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


@register_config('uniform')
@dataclass
class UniformConfig(SamplerConfig):
    """
    Configuration for uniform distribution sampler.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Example:
        config = UniformConfig(min_val=0.01, max_val=0.5)
        sampler = config.create_sampler()
        value = sampler()  # Returns float in [0.01, 0.5]
    """
    min_val: float
    max_val: float

    def validate(self) -> None:
        if self.min_val > self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be <= max_val ({self.max_val})")

    def create_sampler(self) -> Sampler:
        return UniformSampler(self.min_val, self.max_val)

    @classmethod
    def from_dict(cls, config: dict) -> 'UniformConfig':
        """Construct from dict. Required keys: 'min', 'max'."""
        if 'min' not in config or 'max' not in config:
            raise ValueError("UniformConfig requires 'min' and 'max' keys")
        return cls(min_val=config['min'], max_val=config['max'])


@register_config('uniform_int')
@dataclass
class UniformIntConfig(SamplerConfig):
    """
    Configuration for uniform integer distribution sampler.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Example:
        config = UniformIntConfig(min_val=2, max_val=10)
        sampler = config.create_sampler()
        value = sampler()  # Returns int in [2, 10]
    """
    min_val: int
    max_val: int

    def validate(self) -> None:
        if not isinstance(self.min_val, int) or not isinstance(self.max_val, int):
            raise ValueError("min_val and max_val must be integers")
        if self.min_val > self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be <= max_val ({self.max_val})")

    def create_sampler(self) -> Sampler:
        return UniformIntSampler(self.min_val, self.max_val)

    @classmethod
    def from_dict(cls, config: dict) -> 'UniformIntConfig':
        """Construct from dict. Required keys: 'min', 'max'."""
        if 'min' not in config or 'max' not in config:
            raise ValueError("UniformIntConfig requires 'min' and 'max' keys")
        return cls(min_val=int(config['min']), max_val=int(config['max']))


@register_config('meta_gamma')
@dataclass
class GammaConfig(SamplerConfig):
    """
    Configuration for gamma distribution sampler.

    Good for positive values, especially integers like num_layers.
    The alpha and scale parameters are themselves sampled uniformly.

    Args:
        max_alpha: Maximum alpha parameter (shape)
        max_scale: Maximum scale parameter
        lower_bound: Hard lower bound for sampled values
        round_result: If True, round to nearest integer

    Example:
        config = GammaConfig(max_alpha=3, max_scale=2, lower_bound=2, round_result=True)
        sampler = config.create_sampler()
        value = sampler()  # Returns int >= 2
    """
    max_alpha: float
    max_scale: float
    lower_bound: float = 0.0
    round_result: bool = False

    def validate(self) -> None:
        if self.max_alpha <= 0:
            raise ValueError(f"max_alpha ({self.max_alpha}) must be > 0")
        if self.max_scale <= 0:
            raise ValueError(f"max_scale ({self.max_scale}) must be > 0")

    def create_sampler(self) -> Sampler:
        return GammaSampler(self.max_alpha, self.max_scale, self.lower_bound, self.round_result)

    @classmethod
    def from_dict(cls, config: dict) -> 'GammaConfig':
        """Construct from dict. Required keys: 'max_alpha', 'max_scale'."""
        if 'max_alpha' not in config or 'max_scale' not in config:
            raise ValueError("GammaConfig requires 'max_alpha' and 'max_scale' keys")
        return cls(
            max_alpha=config['max_alpha'],
            max_scale=config['max_scale'],
            lower_bound=config.get('lower_bound', 0.0),
            round_result=config.get('round', False)
        )


@register_config('meta_trunc_norm_log_scaled')
@dataclass
class TruncNormLogScaledConfig(SamplerConfig):
    """
    Configuration for truncated normal sampler with log-scaled parameters.

    Good for continuous values that span orders of magnitude
    (e.g., learning rate, noise_std, dropout).

    Args:
        min_mean: Minimum mean value (log-scaled)
        max_mean: Maximum mean value (log-scaled)
        min_std: Minimum std relative to mean
        max_std: Maximum std relative to mean
        lower_bound: Hard lower bound for sampled values
        round_result: If True, round to nearest integer

    Example:
        config = TruncNormLogScaledConfig(min_mean=0.01, max_mean=0.5)
        sampler = config.create_sampler()
        value = sampler()  # Returns float, log-uniformly distributed
    """
    min_mean: float
    max_mean: float
    min_std: float = 0.01
    max_std: float = 1.0
    lower_bound: float = 0.0
    round_result: bool = False

    def validate(self) -> None:
        if self.min_mean <= 0:
            raise ValueError(f"min_mean ({self.min_mean}) must be > 0 (log-scaled)")
        if self.max_mean <= 0:
            raise ValueError(f"max_mean ({self.max_mean}) must be > 0 (log-scaled)")
        if self.min_mean > self.max_mean:
            raise ValueError(f"min_mean ({self.min_mean}) must be <= max_mean ({self.max_mean})")
        if self.min_std <= 0 or self.max_std <= 0:
            raise ValueError("min_std and max_std must be > 0")
        if self.min_std > self.max_std:
            raise ValueError(f"min_std ({self.min_std}) must be <= max_std ({self.max_std})")

    def create_sampler(self) -> Sampler:
        return TruncNormLogScaledSampler(
            self.min_mean, self.max_mean, self.min_std, self.max_std,
            self.lower_bound, self.round_result
        )

    @classmethod
    def from_dict(cls, config: dict) -> 'TruncNormLogScaledConfig':
        """Construct from dict. Required keys: 'min_mean', 'max_mean'."""
        if 'min_mean' not in config or 'max_mean' not in config:
            raise ValueError("TruncNormLogScaledConfig requires 'min_mean' and 'max_mean' keys")
        return cls(
            min_mean=config['min_mean'],
            max_mean=config['max_mean'],
            min_std=config.get('min_std', 0.01),
            max_std=config.get('max_std', 1.0),
            lower_bound=config.get('lower_bound', 0.0),
            round_result=config.get('round', False)
        )


@register_config('meta_beta')
@dataclass
class BetaConfig(SamplerConfig):
    """
    Configuration for beta distribution sampler.

    Good for values in [0, 1] range (e.g., probabilities).
    The beta parameters are themselves sampled uniformly.

    Args:
        min_param: Minimum for beta parameters (a, b)
        max_param: Maximum for beta parameters (a, b)
        scale: Scale factor for output (default 1.0)

    Example:
        config = BetaConfig(min_param=0.5, max_param=2.0, scale=1.0)
        sampler = config.create_sampler()
        value = sampler()  # Returns float in [0, scale]
    """
    min_param: float
    max_param: float
    scale: float = 1.0

    def validate(self) -> None:
        if self.min_param <= 0:
            raise ValueError(f"min_param ({self.min_param}) must be > 0")
        if self.max_param <= 0:
            raise ValueError(f"max_param ({self.max_param}) must be > 0")
        if self.min_param > self.max_param:
            raise ValueError(f"min_param ({self.min_param}) must be <= max_param ({self.max_param})")
        if self.scale <= 0:
            raise ValueError(f"scale ({self.scale}) must be > 0")

    def create_sampler(self) -> Sampler:
        return BetaSampler(self.min_param, self.max_param, self.scale)

    @classmethod
    def from_dict(cls, config: dict) -> 'BetaConfig':
        """Construct from dict. Required keys: 'min', 'max'."""
        if 'min' not in config or 'max' not in config:
            raise ValueError("BetaConfig requires 'min' and 'max' keys")
        return cls(
            min_param=config['min'],
            max_param=config['max'],
            scale=config.get('scale', 1.0)
        )


@register_config('meta_choice')
@dataclass
class ChoiceConfig(SamplerConfig):
    """
    Configuration for categorical choice sampler.

    Weights are sampled randomly, then softmax is applied.
    Good for discrete choices (e.g., activation functions, True/False).

    Args:
        choices: List of possible values to choose from

    Example:
        config = ChoiceConfig(choices=['relu', 'tanh', 'gelu'])
        sampler = config.create_sampler()
        value = sampler()  # Returns one of the choices
    """
    choices: List[Any] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"choices must have at least 2 items, got {len(self.choices)}")

    def create_sampler(self) -> Sampler:
        return ChoiceSampler(self.choices)

    @classmethod
    def from_dict(cls, config: dict) -> 'ChoiceConfig':
        """Construct from dict. Required keys: 'choice_values'."""
        if 'choice_values' not in config:
            raise ValueError("ChoiceConfig requires 'choice_values' key")
        return cls(choices=config['choice_values'])


# =============================================================================
# Sampler Implementations
# =============================================================================

class UniformSampler(Sampler):
    """Sample from uniform distribution."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self) -> float:
        return random.uniform(self.min_val, self.max_val)


class UniformIntSampler(Sampler):
    """Sample integers from uniform distribution."""

    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self) -> int:
        return random.randint(self.min_val, self.max_val)


class GammaSampler(Sampler):
    """Sample from gamma distribution with random parameters."""

    def __init__(self, max_alpha: float, max_scale: float,
                 lower_bound: float = 0, round_result: bool = False):
        self.max_alpha = max_alpha
        self.max_scale = max_scale
        self.lower_bound = lower_bound
        self.round_result = round_result

    def sample(self) -> Union[int, float]:
        # Sample distribution parameters
        log_alpha = random.uniform(0, math.log(self.max_alpha))
        scale = random.uniform(0, self.max_scale)

        alpha = math.exp(log_alpha)
        adjusted_scale = scale / alpha

        # Sample from gamma
        value = self.lower_bound + random.gammavariate(alpha, adjusted_scale)

        if self.round_result:
            return round(value)
        return value


class TruncNormLogScaledSampler(Sampler):
    """Sample from truncated normal with log-scaled mean and std."""

    def __init__(self, min_mean: float, max_mean: float,
                 min_std: float = 0.01, max_std: float = 1.0,
                 lower_bound: float = 0, round_result: bool = False):
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.min_std = min_std
        self.max_std = max_std
        self.lower_bound = lower_bound
        self.round_result = round_result

    def sample(self) -> Union[int, float]:
        # Sample distribution parameters (log-scaled)
        log_mean = random.uniform(math.log(self.min_mean), math.log(self.max_mean))
        log_std = random.uniform(math.log(self.min_std), math.log(self.max_std))

        mean = math.exp(log_mean)
        std = mean * math.exp(log_std)

        # Sample from truncated normal (reject negative values)
        for _ in range(100):
            value = random.gauss(mean, std)
            if value >= 0:
                break
        else:
            value = mean

        value = self.lower_bound + value

        if self.round_result:
            return round(value)
        return value


class BetaSampler(Sampler):
    """Sample from beta distribution with random parameters."""

    def __init__(self, min_param: float, max_param: float, scale: float = 1.0):
        self.min_param = min_param
        self.max_param = max_param
        self.scale = scale

    def sample(self) -> float:
        b = random.uniform(self.min_param, self.max_param)
        k = random.uniform(self.min_param, self.max_param)
        return self.scale * random.betavariate(b, k)


class ChoiceSampler(Sampler):
    """Sample from categorical distribution with random weights."""

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def sample(self) -> Any:
        # Random weights for each choice
        weights = [1.0] + [random.uniform(-3.0, 5.0) for _ in range(len(self.choices) - 1)]
        weights = torch.softmax(torch.tensor(weights), dim=0)

        # Sample from categorical
        idx = torch.multinomial(weights, 1).item()
        return self.choices[idx]


# =============================================================================
# Factory: Create config from dict (using registry)
# =============================================================================

def create_config_from_dict(config_dict: dict) -> SamplerConfig:
    """
    Create a SamplerConfig from a dictionary specification.

    Uses the global registry to find the appropriate config class,
    then calls its from_dict() method to construct the instance.

    Args:
        config_dict: Dict with 'distribution' key and distribution-specific parameters

    Returns:
        SamplerConfig instance

    Raises:
        ValueError: If distribution is unknown or required parameters are missing

    Example:
        config = create_config_from_dict({
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.5
        })
        sampler = config.create_sampler()
    """
    dist = config_dict.get('distribution')

    if dist is None:
        raise ValueError("config_dict must contain 'distribution' key")

    if dist not in _CONFIG_REGISTRY:
        available = list(_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown distribution: '{dist}'. Available: {available}")

    config_class = _CONFIG_REGISTRY[dist]
    return config_class.from_dict(config_dict)


# =============================================================================
# Helper: Check if value is a sampler config dict
# =============================================================================

def _is_sampler_config(value) -> bool:
    """Check if value is a sampler configuration dict."""
    return isinstance(value, dict) and 'distribution' in value


# =============================================================================
# HyperparameterSpace: Collection of fixed values and samplers
# =============================================================================

class HyperparameterSpace:
    """
    A collection of hyperparameters - both fixed values and samplers.

    Automatically detects:
    - Dict with 'distribution' key → create sampler, sample on each call
    - Other values → fixed, return as-is

    Example:
        space = HyperparameterSpace({
            'noise_std': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},  # sampled
            'num_layers': {'distribution': 'meta_gamma', 'max_alpha': 3,        # sampled
                          'max_scale': 2, 'lower_bound': 2, 'round': True},
            'prior_mlp_hidden_dim': 64,           # fixed
            'prior_mlp_activations': 'relu',      # fixed
            'is_causal': True,                    # fixed
        })
        params = space.sample()  # {'noise_std': 0.23, 'num_layers': 4,
                                 #  'prior_mlp_hidden_dim': 64, ...}
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict mapping hyperparameter names to either:
                    - Dict with 'distribution' key → sampled
                    - Any other value → fixed
        """
        self.fixed = {}
        self.samplers = {}

        for name, value in config.items():
            if value is None:
                continue

            if _is_sampler_config(value):
                # Dict with 'distribution' → create sampler
                sampler_config = create_config_from_dict(value)
                self.samplers[name] = sampler_config.create_sampler()
            else:
                # Fixed value
                self.fixed[name] = value

    def sample(self) -> dict:
        """Sample all hyperparameters and return as dict."""
        sampled = {name: sampler.sample() for name, sampler in self.samplers.items()}
        return {**self.fixed, **sampled}


# =============================================================================
# RandomConfigPrior: Main wrapper class
# =============================================================================

class RandomConfigPrior(nn.Module):
    """
    Wraps a data generator and samples random hyperparameters for each batch.

    Hyperparameters can be:
    - Fixed values (int, float, str, bool, etc.)
    - Sampler configs (dict with 'distribution' key)

    Example:
        hyperparameters = {
            # Sampled (dict with 'distribution')
            'noise_std': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
            'num_layers': {'distribution': 'meta_gamma', 'max_alpha': 3,
                          'max_scale': 2, 'lower_bound': 2, 'round': True},
            # Fixed (plain values)
            'prior_mlp_hidden_dim': 64,
            'prior_mlp_activations': 'relu',
            'is_causal': True,
        }
        prior = RandomConfigPrior(mlp.get_batch, hyperparameters, args)
        x, y, y_clean = prior()
    """

    def __init__(self, get_batch, hyperparameters: dict, args: dict):
        """
        Args:
            get_batch: Data generation function (e.g., mlp.get_batch)
            hyperparameters: Dict of hyperparameters (fixed values or sampler configs)
            args: Additional args (device, seq_len, num_features, batch_size)
        """
        super().__init__()
        self.get_batch = get_batch
        self.hp_space = HyperparameterSpace(hyperparameters)
        self.args = args

    def forward(self):
        # Sample hyperparameters (fixed values returned as-is, samplers sampled)
        hyperparameters = self.hp_space.sample()

        # Generate data
        x, y, y_clean = self.get_batch(hyperparameters=hyperparameters, **self.args)

        return x, y, y_clean


# =============================================================================
# Module-level get_batch function
# =============================================================================

@torch.no_grad()
def get_batch(batch_size: int, seq_len: int, num_features: int, get_batch,
              device, hyperparameters: Optional[dict] = None,
              batch_size_per_gp_sample: Optional[int] = None, **kwargs):
    """
    Generate batches with randomly sampled hyperparameters.

    Hyperparameters can be:
    - Fixed values: int, float, str, bool, list, tuple, etc.
    - Sampler configs: dict with 'distribution' key

    Args:
        batch_size: Total batch size
        seq_len: Sequence length (number of samples per batch)
        num_features: Number of features
        get_batch: Underlying data generator (e.g., mlp.get_batch)
        device: torch device
        hyperparameters: Dict of hyperparameters. Values can be:
            - Plain values (fixed)
            - Dict with 'distribution' key (sampled per sub-batch)
        batch_size_per_gp_sample: Sub-batch size (samples sharing same config)

    Returns:
        x: Features, shape (seq_len, batch_size, num_features)
        y: Targets, shape (seq_len, batch_size)
        y_clean: Clean targets (before noise), shape (seq_len, batch_size)

    Example:
        x, y, y_ = get_batch(
            batch_size=32,
            seq_len=100,
            num_features=20,
            get_batch=mlp.get_batch,
            device='cuda',
            hyperparameters={
                # Sampled hyperparameters (dict with 'distribution')
                'noise_std': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
                'num_layers': {'distribution': 'meta_gamma', 'max_alpha': 3,
                              'max_scale': 2, 'lower_bound': 2, 'round': True},
                # Fixed hyperparameters (plain values)
                'prior_mlp_hidden_dim': 64,
                'prior_mlp_activations': 'relu',
                'is_causal': True,
            },
            batch_size_per_gp_sample=8  # 4 different random configs
        )
    """
    hyperparameters = hyperparameters or {}

    batch_size_per_gp_sample = batch_size_per_gp_sample or min(64, batch_size)
    num_models = batch_size // batch_size_per_gp_sample

    assert num_models > 0, \
        f'Batch size ({batch_size}) too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    assert num_models * batch_size_per_gp_sample == batch_size, \
        f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    args = {
        'device': device,
        'seq_len': seq_len,
        'num_features': num_features,
        'batch_size': batch_size_per_gp_sample,
        **kwargs
    }

    # Create models with different random configs
    models = [
        RandomConfigPrior(get_batch, hyperparameters, args)
        for _ in range(num_models)
    ]

    # Generate samples
    samples = [model() for model in models]

    # Concatenate results
    x, y, y_clean = zip(*samples)
    x = torch.cat(x, dim=1).detach()
    y = torch.cat(y, dim=1).detach()
    y_clean = torch.cat(y_clean, dim=1).detach()

    return x, y, y_clean


DataLoader = get_batch_to_dataloader(get_batch)
