# Synthetic Tabular Data with Causal Structure

![img](./_icon.jpg)
Tab Prior is a lightweight library extracted from TabPFN that focuses entirely on building rich synthetic tabular datasets. It is designed for researchers who need controllable data with correlated features, customizable causal structures, and configurable noise processes to benchmark tabular models or explore counterfactual scenarios.

## Why Tab Prior?
- **Causal-aware generation** – `prior/mlp.py` can wire features through random DAGs so that only a subset of nodes cause the target, letting you test algorithms that reason about conditional dependencies.
- **Correlated features by design** – mix Gaussian processes (`prior/fast_gp.py`) and neural generators to obtain smooth, multi-modal, or highly collinear feature spaces.
- **Flexible observation models** – `prior/flexible_categorical.py` adds ranking transforms, categorical projections, and controllable missingness patterns that resemble real-world tabular datasets.
- **Differentiable prior bag** – `prior/differentiable_prior.py` samples hyperparameters on the fly so every batch can come from a different synthetic world, ideal for meta-learning.
- **Drop-in dataloaders** – utilities in `prior/utils.py` turn any `get_batch` function into a PyTorch `DataLoader`, including curriculum control over sequence lengths.

## Installation
Install directly from PyPI:
```bash
pip install tab-prior
```
This installs the core dependencies (`torch`, `numpy`, `scipy`, `gpytorch`). Plotting extras can be added with `pip install tab-prior[plotting]`.

For local development instead, clone the repo and run:
```bash
pip install -e .
```

## Repository Layout
- `prior/mlp.py` – neural generator with causal graphs, dropout noise, and feature permutation utilities.
- `prior/fast_gp.py` – fast Gaussian process sampler for smooth latent functions.
- `prior/flexible_categorical.py` – wraps any generator to produce regression or classification tasks with controllable missing values and categorical leakage.
- `prior/prior_bag.py` – mixture model that samples between multiple generators per batch.
- `prior/differentiable_prior.py` – framework for defining distributions over hyperparameters (uniform, gamma, truncated-normal, categorical, etc.).
- `prior/utils.py` – helper math ops, ranking transforms, and `get_batch_to_dataloader` for training loops.
- `example/` – runnable scripts that demonstrate each prior and serve as smoke tests.


## Example Scripts & Tests
The `example/` directory contains runnable snippets such as `test_mlp_prior.py`, `test_gp_prior.py`, `test_prior_bag.py`, and `guide_differentiable_prior.py`. Run any of them with:
```bash
python example/test_mlp_prior.py
```
They print tensor shapes, sample statistics, and demonstrate typical hyperparameter dictionaries.
