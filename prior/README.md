# TabPFN Priors: Standalone Synthetic Data Generator Tutorial

## Overview

The `tabpfn.priors` package is a powerful synthetic tabular data generator that can be decoupled from TabPFN and used in any project. It generates realistic synthetic datasets on-the-fly with diverse characteristics.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Advanced Usage](#advanced-usage)

---

## Dependencies

### Required Packages
```bash
pip install torch numpy scipy gpytorch
```

### Minimal Utils Module

The priors package needs a few utilities from `tabpfn.utils`. Create a `minimal_utils.py`:

```python
import torch
import random

# Device detection
default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

def set_locals_in_self(locals_dict):
    """Set all local variables as object attributes."""
    self = locals_dict['self']
    for var_name, val in locals_dict.items():
        if var_name != 'self':
            setattr(self, var_name, val)

def torch_nanmean(x, dim=0, return_nanshare=False):
    """Calculate mean ignoring NaN values."""
    mask = ~torch.isnan(x)
    num = mask.sum(dim=dim)
    value = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
    mean = value / num
    if return_nanshare:
        return mean, 1. - num / x.shape[dim]
    return mean

def torch_nanstd(x, dim=0):
    """Calculate std ignoring NaN values."""
    mask = ~torch.isnan(x)
    num = mask.sum(dim=dim)
    value = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
    mean = value / num
    mean_broadcast = mean.unsqueeze(dim).expand_as(x)
    quadratic_diff = torch.where(mask, (mean_broadcast - x) ** 2, torch.zeros_like(x))
    return torch.sqrt(quadratic_diff.sum(dim=dim) / (num - 1))

def normalize_data(data, normalize_positions=-1):
    """Normalize data to zero mean and unit variance."""
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + 1e-6
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + 1e-6
    data = (data - mean) / std
    return torch.clip(data, min=-100, max=100)

def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    """Remove outliers using n-sigma clipping."""
    assert len(X.shape) == 3, "X must be T,B,H"
    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean = torch_nanmean(data, dim=0)
    data_std = torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    # Recalculate with mask
    num = mask.sum(dim=0)
    masked_data = torch.where(mask, data, torch.zeros_like(data))
    data_mean = masked_data.sum(dim=0) / num

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X

def to_ranking_low_mem(data):
    """Convert data to rankings (low memory version)."""
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = (data[:, :, col] >= data[:, :, col].unsqueeze(-2))
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x

def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    """Normalize by the number of features actually used."""
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** 0.5
    return x / (num_features_used / num_features)

def get_nan_value(v, set_value_to_nan=0.0):
    """Get NaN replacement value."""
    if random.random() < set_value_to_nan:
        return v
    return random.choice([-999, 0, 1, 999])

def nan_handling_missing_for_unknown_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('nan'), set_value_to_nan)

def nan_handling_missing_for_no_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('-inf'), set_value_to_nan)

def nan_handling_missing_for_a_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('inf'), set_value_to_nan)
```

---

## Quick Start

### Example 1: Generate MLP-based Synthetic Data

```python
import torch
from tabpfn.priors import mlp

# Configuration
hyperparameters = {
    'num_layers': 3,
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': True,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.2,
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
    'new_mlp_per_example': False
}

# Generate batch
batch_size = 32
seq_len = 100  # number of samples
num_features = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x, y, y_clean = mlp.get_batch(
    batch_size=batch_size,
    seq_len=seq_len,
    num_features=num_features,
    hyperparameters=hyperparameters,
    device=device,
    num_outputs=1
)

print(f"Features shape: {x.shape}")  # (seq_len, batch_size, num_features)
print(f"Targets shape: {y.shape}")   # (seq_len, batch_size)
print(f"Features range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Targets range: [{y.min():.2f}, {y.max():.2f}]")
```

### Example 2: Generate GP-based Synthetic Data

```python
from tabpfn.priors import fast_gp

# GP hyperparameters
hyperparameters = {
    'noise': 0.1,
    'outputscale': 1.0,
    'lengthscale': 0.5,
    'sampling': 'uniform',
    'fast_computations': (True, True, True)
}

# Generate smooth functions using Gaussian Process
x, y, y_clean = fast_gp.get_batch(
    batch_size=16,
    seq_len=50,
    num_features=5,
    device=device,
    hyperparameters=hyperparameters
)

print(f"GP Features shape: {x.shape}")
print(f"GP Targets shape: {y.shape}")
```

### Example 3: Generate Classification Data with Flexible Categorical

```python
from tabpfn.priors import flexible_categorical, mlp

# Classification configuration
hyperparameters = {
    # MLP configuration
    'num_layers': 3,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.1,
    'y_is_effect': False,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.Tanh(),
    'block_wise_dropout': False,
    'prior_mlp_scale_weights_sqrt': True,
    'init_std': 1.0,
    'mix_activations': False,
    'sort_features': False,
    'in_clique': False,
    'random_feature_rotation': False,
    'sampling': 'normal',
    'new_mlp_per_example': False,

    # Classification wrapper config
    'num_classes': 3,  # For 3-class classification
    'balanced': False,
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.5,
    'nan_prob_no_reason': 0.1,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'nan_prob_unknown_reason_reason_prior': 0.5,
    'categorical_feature_p': 0.2,
    'normalize_to_ranking': False,
    'set_value_to_nan': 0.2,
    'normalize_by_used_features': True,
    'num_features_used': 15,
    'seq_len_used': 80,
    'normalize_labels': True,
    'check_is_compatible': False,
    'normalize_ignore_label_too': False,
    'rotate_normalized_labels': True
}

# Generate classification data
x, y, y_clean = flexible_categorical.get_batch(
    batch_size=32,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device=device,
    hyperparameters=hyperparameters,
    batch_size_per_gp_sample=8,
    single_eval_pos=80
)

print(f"Classification features shape: {x.shape}")
print(f"Classification targets shape: {y.shape}")
print(f"Unique classes: {torch.unique(y)}")
```

---

## API Reference

### 1. MLP Prior (`mlp.get_batch`)

Generates synthetic data using randomly initialized neural networks.

**Parameters:**
- `batch_size` (int): Number of independent datasets
- `seq_len` (int): Number of samples per dataset
- `num_features` (int): Number of input features
- `hyperparameters` (dict): Configuration dictionary
- `device` (str): 'cuda' or 'cpu'
- `num_outputs` (int): Number of output dimensions

**Key Hyperparameters:**
```python
{
    'num_layers': int,              # Network depth (≥2)
    'prior_mlp_hidden_dim': int,    # Hidden layer size
    'is_causal': bool,              # Use causal graph structure
    'num_causes': int,              # Number of causal variables
    'noise_std': float,             # Gaussian noise level
    'prior_mlp_dropout_prob': float, # Dropout probability
    'sampling': str,                # 'normal', 'uniform', or 'mixed'
    'prior_mlp_activations': callable # Activation function factory
}
```

**Returns:**
- `x`: Features tensor of shape `(seq_len, batch_size, num_features)`
- `y`: Targets tensor of shape `(seq_len, batch_size)`
- `y_clean`: Same as `y` (no noise distinction in current version)

---

### 2. Gaussian Process Prior (`fast_gp.get_batch`)

Generates smooth functions using Gaussian Processes.

**Parameters:**
- `batch_size` (int): Number of independent GPs
- `seq_len` (int): Number of sample points
- `num_features` (int): Input dimensionality
- `device` (str): Device
- `hyperparameters` (dict): GP configuration

**Key Hyperparameters:**
```python
{
    'noise': float,              # Observation noise
    'outputscale': float,        # Kernel output scale
    'lengthscale': float,        # RBF kernel lengthscale
    'sampling': str,             # 'uniform' or 'normal' input sampling
    'fast_computations': tuple   # GPyTorch optimization flags
}
```

**Returns:**
- `x`: Input locations `(seq_len, batch_size, num_features)`
- `y`: GP samples `(seq_len, batch_size)`
- `y_clean`: Same as `y`

---

### 3. Flexible Categorical (`flexible_categorical.get_batch`)

Wrapper that adds classification/regression transformations and data augmentation.

**Parameters:**
- `batch_size` (int): Total batch size
- `seq_len` (int): Sequence length
- `num_features` (int): Number of features
- `get_batch` (callable): Underlying prior (e.g., `mlp.get_batch`)
- `device` (str): Device
- `hyperparameters` (dict): Full configuration
- `batch_size_per_gp_sample` (int): Batch subdivision size

**Key Hyperparameters:**
```python
{
    # Base prior config (passed through)
    ...mlp or gp hyperparameters...,

    # Classification/Regression
    'num_classes': int,              # 0=regression, 2+=classification
    'balanced': bool,                # Balance classes
    'multiclass_type': str,          # 'rank', 'value', or 'multi_node'
    'output_multiclass_ordered_p': float,  # Probability of ordered classes

    # Missing data injection
    'nan_prob_no_reason': float,     # Random missing data probability
    'nan_prob_a_reason': float,      # Structured missing data
    'nan_prob_unknown_reason': float,# Unknown missingness mechanism
    'set_value_to_nan': float,       # Use actual NaN vs sentinel values

    # Feature engineering
    'categorical_feature_p': float,  # Probability of categorical features
    'normalize_to_ranking': bool,    # Use ranking transformation
    'num_features_used': int,        # Number of informative features
    'normalize_by_used_features': bool,

    # Label processing
    'normalize_labels': bool,        # Remap labels to 0,1,2,...
    'check_is_compatible': bool      # Ensure train/test class overlap
}
```

---

### 4. Prior Bag (`prior_bag.get_batch`)

Mixture of multiple priors sampled dynamically.

**Parameters:**
- `batch_size` (int)
- `seq_len` (int)
- `num_features` (int)
- `device` (str)
- `hyperparameters` (dict): Must include `prior_bag_get_batch` and weights

**Key Hyperparameters:**
```python
{
    'prior_bag_get_batch': [callable_1, callable_2, ...],  # List of priors
    'prior_bag_exp_weights_1': float,  # Exponential weight for prior 1
    'prior_bag_exp_weights_2': float,  # Exponential weight for prior 2
    # ... (first prior always has weight 1.0)
}
```

---

## Examples

### Example 4: Using DataLoader for Training

```python
from tabpfn.priors.utils import get_batch_to_dataloader
from tabpfn.priors import mlp

# Create DataLoader class
DataLoader = get_batch_to_dataloader(mlp.get_batch)

# Hyperparameters
hyperparameters = {
    'num_layers': 3,
    'prior_mlp_hidden_dim': 64,
    'is_causal': True,
    'num_causes': 8,
    'noise_std': 0.1,
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
    'new_mlp_per_example': False
}

# Evaluation position sampler
def eval_pos_seq_len_sampler():
    eval_pos = 80  # Use first 80 for training, rest for evaluation
    seq_len = 100
    return eval_pos, seq_len

# Create dataloader
dl = DataLoader(
    num_steps=50,  # Steps per epoch
    batch_size=64,
    num_features=20,
    hyperparameters=hyperparameters,
    device='cuda',
    eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
    seq_len_maximum=100
)

# Use in training loop
class DummyModel:
    pass

dl.model = DummyModel()  # Required before iteration

for epoch in range(3):
    for batch_idx, ((style, x, y), target_y, single_eval_pos) in enumerate(dl):
        # style: hyperparameter embeddings (or None)
        # x: features (seq_len, batch_size, num_features)
        # y: targets (seq_len, batch_size)
        # target_y: same as y
        # single_eval_pos: where evaluation starts

        print(f"Epoch {epoch}, Batch {batch_idx}")
        print(f"  Train data: x[:{single_eval_pos}], y[:{single_eval_pos}]")
        print(f"  Eval data: x[{single_eval_pos}:], y[{single_eval_pos}:]")

        # Your training code here
        break  # Just show first batch
    break
```

### Example 5: Mixed GP and MLP Prior (Prior Bag)

```python
from tabpfn.priors import prior_bag, mlp, fast_gp

# Define hyperparameters for both priors
base_hparams = {
    # MLP params
    'num_layers': 3,
    'prior_mlp_hidden_dim': 32,
    'is_causal': True,
    'num_causes': 8,
    'noise_std': 0.1,
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

    # GP params
    'noise': 0.1,
    'outputscale': 1.0,
    'lengthscale': 0.5,
    'fast_computations': (True, True, True),

    # Prior bag weights (softmax applied)
    'prior_bag_get_batch': [fast_gp.get_batch, mlp.get_batch],
    'prior_bag_exp_weights_1': 5.0,  # Higher weight = more likely to sample MLP
    'verbose': True  # See which prior is selected
}

# Generate mixed data
x, y, y_clean = prior_bag.get_batch(
    batch_size=32,
    seq_len=100,
    num_features=10,
    device='cuda',
    hyperparameters=base_hparams,
    batch_size_per_gp_sample=8
)

print(f"Mixed prior output: {x.shape}")
```

### Example 6: Binary Classification with Missing Data

```python
from tabpfn.priors import flexible_categorical, mlp

config = {
    # MLP base
    'num_layers': 2,
    'prior_mlp_hidden_dim': 32,
    'is_causal': False,
    'num_causes': 10,
    'noise_std': 0.05,
    'y_is_effect': False,
    'pre_sample_weights': True,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': True,
    'prior_mlp_activations': lambda: torch.nn.Tanh(),
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
    'balanced': True,  # Balanced binary classification
    'multiclass_type': 'rank',
    'output_multiclass_ordered_p': 0.0,

    # Add 20% missing data
    'nan_prob_no_reason': 0.2,
    'nan_prob_a_reason': 0.0,
    'nan_prob_unknown_reason': 0.0,
    'set_value_to_nan': 1.0,  # Use actual NaN values

    # 30% of features are categorical
    'categorical_feature_p': 0.3,

    'normalize_to_ranking': False,
    'normalize_by_used_features': True,
    'num_features_used': 15,
    'seq_len_used': 100,
    'normalize_labels': False,
    'check_is_compatible': False
}

x, y, _ = flexible_categorical.get_batch(
    batch_size=16,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device='cpu',
    hyperparameters=config
)

print(f"Binary classification with missing data:")
print(f"  Shape: {x.shape}")
print(f"  Missing values: {torch.isnan(x).sum().item()} / {x.numel()}")
print(f"  Class distribution: {torch.bincount(y.long().flatten())}")
```

---

## Advanced Usage

### Differentiable Hyperparameters

For meta-learning or hyperparameter optimization, use `DifferentiableHyperparameter`:

```python
from tabpfn.priors.differentiable_prior import DifferentiableHyperparameter, get_batch

# Define differentiable hyperparameters
diff_hparams = {
    'num_layers': {
        'distribution': 'meta_gamma',
        'max_alpha': 2,
        'max_scale': 3,
        'round': True,
        'lower_bound': 2
    },
    'noise_std': {
        'distribution': 'meta_trunc_norm_log_scaled',
        'max_mean': 0.3,
        'min_mean': 0.0001,
        'round': False,
        'lower_bound': 0.0
    },
    'is_causal': {
        'distribution': 'meta_choice',
        'choice_values': [True, False]
    }
}

# Fixed hyperparameters
fixed_hparams = {
    'prior_mlp_hidden_dim': 64,
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
    'emsize': 64  # Required for differentiable hparams
}

# Generate with sampled hyperparameters
x, y, y_clean, hp_indicators = get_batch(
    batch_size=32,
    seq_len=100,
    num_features=20,
    get_batch=mlp.get_batch,
    device='cuda',
    differentiable_hyperparameters=diff_hparams,
    hyperparameters=fixed_hparams
)

print(f"Hyperparameter indicators shape: {hp_indicators.shape if hp_indicators is not None else None}")
```

---

## Tips and Best Practices

1. **Start Simple**: Begin with basic MLP or GP priors before using flexible_categorical
2. **Batch Size**: Use `batch_size_per_gp_sample` to control memory usage
3. **Device Management**: Always specify device explicitly for consistent behavior
4. **NaN Handling**: When using missing data, ensure your model can handle NaN values
5. **Hyperparameter Tuning**: Start with default values and adjust incrementally
6. **Causal Mode**: Use `is_causal=True` for more realistic feature dependencies
7. **Memory**: GP prior can be memory-intensive for long sequences; use `fast_computations`

---

## Common Issues

**Issue**: Out of memory
- **Solution**: Reduce `batch_size`, `seq_len`, or use `batch_size_per_gp_sample`

**Issue**: NaN in generated data
- **Solution**: Reduce `noise_std`, check `init_std`, or disable problematic transformations

**Issue**: All same class in classification
- **Solution**: Set `balanced=True` or adjust `output_multiclass_ordered_p`

**Issue**: Import errors
- **Solution**: Ensure minimal_utils.py is in your path and update imports in priors to use it

---

## Integration Checklist

To use priors standalone:

- [ ] Copy `tabpfn/priors/` directory to your project
- [ ] Create `minimal_utils.py` with required utilities
- [ ] Update imports in priors files: `from tabpfn.utils import X` → `from minimal_utils import X`
- [ ] Install dependencies: `torch`, `numpy`, `scipy`, `gpytorch`
- [ ] Test with Quick Start examples
- [ ] Adapt hyperparameters for your use case

---

## Citation

If you use this in your research, please cite the TabPFN paper:

```bibtex
@article{hollmann2022tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  journal={arXiv preprint arXiv:2207.01848},
  year={2022}
}
```
