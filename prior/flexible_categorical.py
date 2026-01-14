import random

import torch
from torch import nn

from .utils import get_batch_to_dataloader
from .utils import (
    normalize_data, to_ranking_low_mem, remove_outliers, normalize_by_used_features_f,
    nan_handling_missing_for_unknown_reason_value, nan_handling_missing_for_no_reason_value,
    nan_handling_missing_for_a_reason_value, randomize_classes, CategoricalActivation,
    uniform_int_sampler_f
)


# =============================================================================
# Class Assigners: Convert regression targets to classification labels
# =============================================================================

class BalancedBinarize(nn.Module):
    """Binary classification with 50/50 split at median."""
    def forward(self, x):
        return (x > torch.median(x)).float()


class RegressionNormalized(nn.Module):
    """Keep as regression, normalize to [0, 1] range."""
    def forward(self, x):
        maxima = torch.max(x, 0)[0]
        minima = torch.min(x, 0)[0]
        return (x - minima) / (maxima - minima)


def _sample_num_classes(min_classes, max_classes):
    """Sample number of classes, biased towards 2."""
    if random.random() > 0.5:
        return uniform_int_sampler_f(min_classes, max_classes)()
    return 2


class MulticlassRank(nn.Module):
    """
    Assign classes based on rank thresholds.

    Randomly samples data points as class boundaries, then assigns classes
    based on how many boundaries each value exceeds.

    Example:
        values:     [0.1, 0.5, 0.3, 0.9, 0.7]  (regression targets)
        boundaries: [0.3, 0.7]                  (sampled from data)
        classes:    [0,   1,   0,   2,   1]    (count of exceeded boundaries)

        If ordered_p=0.0, classes may be shuffled: [2, 0, 2, 1, 0]
    """
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = _sample_num_classes(2, num_classes)
        self.ordered_p = ordered_p

    def forward(self, x):
        # Sample random positions as class boundaries
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)
        d = (x > class_boundaries).sum(axis=0)

        # Randomly shuffle classes for some batches
        randomized = torch.rand((d.shape[1],)) > self.ordered_p
        d[:, randomized] = randomize_classes(d[:, randomized], self.num_classes)

        # Randomly reverse class order for some batches
        reverse = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse] = self.num_classes - 1 - d[:, reverse]
        return d


class MulticlassValue(nn.Module):
    """
    Assign classes based on fixed random thresholds.

    Uses pre-sampled random thresholds (from standard normal) as class
    boundaries. Unlike MulticlassRank, thresholds are independent of data.

    Example:
        values:     [0.1, 0.5, -0.3, 1.2, 0.8]  (regression targets)
        thresholds: [0.0, 0.6]                   (fixed, sampled from N(0,1))
        classes:    [1,   1,    0,   2,   2]    (count of exceeded thresholds)

        If ordered_p=0.0, classes may be shuffled: [0, 0, 2, 1, 1]
    """
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = _sample_num_classes(2, num_classes)
        self.classes = nn.Parameter(torch.randn(self.num_classes - 1), requires_grad=False)
        self.ordered_p = ordered_p

    def forward(self, x):
        d = (x > self.classes.unsqueeze(-1).unsqueeze(-1)).sum(axis=0)

        randomized = torch.rand((d.shape[1],)) > self.ordered_p
        d[:, randomized] = randomize_classes(d[:, randomized], self.num_classes)

        reverse = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse] = self.num_classes - 1 - d[:, reverse]
        return d


class MulticlassMultiNode(nn.Module):
    """
    Assign classes using softmax-based multinomial sampling.

    Requires 3D input (seq_len, batch, num_classes). Applies sigmoid to first
    num_classes features, then samples class from multinomial distribution.
    Falls back to MulticlassValue for 2D input.

    Example:
        x shape: (100, 8, 5)  with num_classes=3
        logits:  x[:, :, :3]  -> shape (100, 8, 3)
        probs:   sigmoid(logits) ** 3  (temperature scaling)
        classes: multinomial(probs)    -> shape (100, 8), values in {0, 1, 2}

        This creates soft, probabilistic class assignments based on feature values.
    """
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = _sample_num_classes(2, num_classes)
        self.fallback = MulticlassValue(num_classes, ordered_p)

    def forward(self, x):
        if len(x.shape) == 2:
            return self.fallback(x)
        x = x.clone()
        x[torch.isnan(x)] = 1e-5
        probs = torch.sigmoid(x[:, :, :self.num_classes]).reshape(-1, self.num_classes)
        probs = torch.pow(probs + 1e-5, 3)  # temperature=3
        d = torch.multinomial(probs, 1, replacement=True)
        return d.reshape(x.shape[0], x.shape[1])


def _create_class_assigner(num_classes, balanced, multiclass_type, ordered_p):
    """
    Factory function to create the appropriate class assigner.

    Args:
        num_classes: 
            Number of classes (0=regression, 2=binary, 3+=multiclass)
        balanced: 
            If True, use median split for binary (50/50 class balance)
        multiclass_type: 
            One of 'rank', 'value', 'multi_node'
        ordered_p: 
            Probability of preserving value ordering in class labels
                   (1.0=ordinal, 0.0=nominal/shuffled)
            
            ordered (ordered_p=1.0):   classes = [0, 1, 2]  (value order preserved)
            shuffled (ordered_p=0.0):  classes = [2, 0, 1]  (random permutation)

            This simulates both:
            - Ordinal classification: e.g., ratings (1-5 stars), grades (A/B/C)
            - Nominal classification: e.g., categories (cat/dog/bird) with no inherent order
    Returns:
        nn.Module that converts regression targets to class labels

    Decision tree:
        num_classes=0                    -> RegressionNormalized (keep as regression)
        num_classes=2, balanced=True     -> BalancedBinarize (median split)
        num_classes>=2, balanced=False   -> MulticlassRank/Value/MultiNode
    """
    if num_classes == 0:
        return RegressionNormalized()

    if num_classes == 2 and balanced:
        return BalancedBinarize()

    if num_classes > 2 and balanced:
        raise NotImplementedError("Balanced multiclass (>2 classes) is not supported")

    # Unbalanced multiclass
    assigners = {
        'rank': lambda: MulticlassRank(num_classes, ordered_p),
        'value': lambda: MulticlassValue(num_classes, ordered_p),
        'multi_node': lambda: MulticlassMultiNode(num_classes, ordered_p),
    }
    if multiclass_type not in assigners:
        raise ValueError(f"Unknown multiclass_type: {multiclass_type}")
    return assigners[multiclass_type]()


# =============================================================================
# FlexibleCategorical: Post-processing wrapper for data generation
# =============================================================================

class FlexibleCategorical(torch.nn.Module):
    """
    Wraps a data generator (MLP/GP) and applies post-processing:
    1. Add missing values (MCAR, MAR, or unknown)
    2. Convert some features to categorical
    3. Normalize features
    4. Convert regression targets to classification
    5. Pad with zero features
    6. Normalize labels
    """

    def __init__(self, get_batch, hyperparameters, args):
        super().__init__()
        self.h = hyperparameters
        self.args = args
        self.get_batch = get_batch

        self._validate_hyperparameters()

        self.class_assigner = _create_class_assigner(
            num_classes=self.h['num_classes'],
            balanced=self.h['balanced'],
            multiclass_type=self.h.get('multiclass_type', 'rank'),
            ordered_p=self.h.get('output_multiclass_ordered_p', 0.5)
        )

    def _validate_hyperparameters(self):
        """
        Validate hyperparameters and check for conflicts.

        Hyperparameter Tutorial:

        1. TASK TYPE (num_classes, balanced)
           num_classes=0              -> Regression
           num_classes=2, balanced    -> Binary classification (50/50 split)
           num_classes=3+             -> Multiclass classification

        2. MULTICLASS METHOD (multiclass_type, output_multiclass_ordered_p)
           multiclass_type='rank'     -> Class by rank position in data
           multiclass_type='value'    -> Class by fixed thresholds
           multiclass_type='multi_node' -> Class by softmax sampling
           output_multiclass_ordered_p -> 1.0=ordinal, 0.0=nominal/shuffled

        3. MISSING VALUES (nan_prob_*, set_value_to_nan)
           nan_prob_no_reason         -> MCAR: random missing
           nan_prob_a_reason          -> MAR: structured missing
           nan_prob_unknown_reason    -> Mix of MCAR/MAR
           set_value_to_nan           -> 1.0=NaN, 0.0=sentinel (-999,0,1,999)

        4. FEATURES (num_features_used, categorical_feature_p)
           num_features_used          -> Informative features (rest are zero-padded)
           categorical_feature_p      -> Probability of categorical conversion

        5. NORMALIZATION (normalize_to_ranking, normalize_labels)
           normalize_to_ranking       -> True=rank transform, False=z-score
           normalize_labels           -> Remap labels to 0,1,2,...
           rotate_normalized_labels   -> Randomly shift label indices

        Example minimal config:
            {
                'num_classes': 2, 'balanced': True,
                'nan_prob_no_reason': 0.0, 'nan_prob_a_reason': 0.0,
                'nan_prob_unknown_reason': 0.0, 'set_value_to_nan': 1.0,
                'normalize_to_ranking': False, 'num_features_used': 10,
            }
        """
        h = self.h
        args = self.args

        # --- Required parameters ---
        required = ['num_classes', 'balanced', 'nan_prob_no_reason',
                    'nan_prob_a_reason', 'nan_prob_unknown_reason',
                    'set_value_to_nan', 'normalize_to_ranking', 'num_features_used']
        missing = [k for k in required if k not in h]
        if missing:
            raise ValueError(f"Missing required hyperparameters: {missing}")

        # --- Classification validation ---
        if h['num_classes'] < 0:
            raise ValueError(f"num_classes must be >= 0, got {h['num_classes']}")

        if h['num_classes'] > 2 and h['balanced']:
            raise ValueError(
                f"balanced=True only works with binary classification (num_classes=2), "
                f"but got num_classes={h['num_classes']}"
            )

        if h['num_classes'] > 1 and not h['balanced']:
            valid_types = ['rank', 'value', 'multi_node']
            mtype = h.get('multiclass_type', 'rank')
            if mtype not in valid_types:
                raise ValueError(f"multiclass_type must be one of {valid_types}, got '{mtype}'")

        # --- Probability validation ---
        prob_params = ['nan_prob_no_reason', 'nan_prob_a_reason', 'nan_prob_unknown_reason',
                       'set_value_to_nan', 'output_multiclass_ordered_p', 'categorical_feature_p']
        for param in prob_params:
            if param in h:
                val = h[param]
                if not (0 <= val <= 1):
                    raise ValueError(f"{param} must be in [0, 1], got {val}")

        # --- Feature validation ---
        num_features_used = h['num_features_used']
        num_features = args.get('num_features', num_features_used)

        if num_features_used <= 0:
            raise ValueError(f"num_features_used must be > 0, got {num_features_used}")

        if num_features_used > num_features:
            raise ValueError(
                f"num_features_used ({num_features_used}) cannot exceed "
                f"num_features ({num_features})"
            )

        # --- Warn about potential issues ---
        if h.get('check_is_compatible', False) and 'single_eval_pos' not in args:
            import warnings
            warnings.warn(
                "check_is_compatible=True but single_eval_pos not provided. "
                "Will use seq_len // 2 as default."
            )

    # -------------------------------------------------------------------------
    # Step 1: Generate raw data
    # -------------------------------------------------------------------------
    def _generate_raw_data(self):
        """Get raw data from underlying generator."""
        args = {**self.args, 'num_features': self.h['num_features_used']}
        return self.get_batch(hyperparameters=self.h, **args)

    # -------------------------------------------------------------------------
    # Step 2: Add missing values
    # -------------------------------------------------------------------------
    def _add_missing_values(self, x):
        """
        Inject missing values with different patterns.

        Missing value representation controlled by 'set_value_to_nan':
            1.0 -> Always use NaN
            0.0 -> Always use sentinel values (-999, 0, 1, 999)
            0.7 -> 70% NaN, 30% sentinel values

        Note: Only NaN values are detected by torch.isnan(). Sentinel values
        won't show up as missing in stats like torch.isnan(x).sum().
        """
        h = self.h
        total_nan_prob = h['nan_prob_no_reason'] + h['nan_prob_a_reason'] + h['nan_prob_unknown_reason']

        # Only apply to ~50% of batches
        if total_nan_prob <= 0 or random.random() > 0.5:
            return x

        # MCAR: Missing Completely At Random
        if random.random() < h['nan_prob_no_reason']:
            value = nan_handling_missing_for_no_reason_value(h['set_value_to_nan'])
            x = self._drop_random(x, value, h['nan_prob_no_reason'])

        # MAR: Missing At Random (structured)
        if h['nan_prob_a_reason'] > 0 and random.random() > 0.5:
            value = nan_handling_missing_for_a_reason_value(h['set_value_to_nan'])
            x = self._drop_structured(x, value)

        # Unknown: Could be either MCAR or MAR
        if h['nan_prob_unknown_reason'] > 0:
            value = nan_handling_missing_for_unknown_reason_value(h['set_value_to_nan'])
            if random.random() < h.get('nan_prob_unknown_reason_reason_prior', 0.5):
                x = self._drop_random(x, value, h['nan_prob_no_reason'])
            else:
                x = self._drop_structured(x, value)

        return x

    def _drop_random(self, x, value, prob):
        """MCAR: Drop values completely at random."""
        mask = torch.rand(x.shape, device=x.device) < random.random() * prob
        # calculate mask number
        x[mask] = value
        return x

    def _drop_structured(self, x, value):
        """MAR: Drop values based on data patterns."""
        sampler = CategoricalActivation(
            ordered_p=0.0, categorical_p=1.0,
            keep_activation_size=False, num_classes_sampler=lambda: 20
        )
        pattern = sampler(x)
        threshold = torch.rand((1,), device=x.device) * 20 * self.h['nan_prob_no_reason'] * random.random()
        x[pattern < threshold] = value
        return x

    # -------------------------------------------------------------------------
    # Step 3: Convert features to categorical
    # -------------------------------------------------------------------------
    def _make_categorical_features(self, x):
        """Convert some continuous features to categorical."""
        prob = self.h.get('categorical_feature_p', 0)
        if prob <= 0 or random.random() >= prob:
            return x

        col_prob = random.random()  # probability for each column
        for col in range(x.shape[2]):
            if random.random() < col_prob:
                num_categories = max(round(random.gammavariate(1, 10)), 2)
                discretizer = MulticlassRank(num_categories, ordered_p=0.3)
                x[:, :, col] = discretizer(x[:, :, col])
        return x

    # -------------------------------------------------------------------------
    # Step 4: Normalize features
    # -------------------------------------------------------------------------
    def _normalize_features(self, x, y):
        """Normalize features using ranking or z-score."""
        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)
        return normalize_data(x), normalize_data(y)

    # -------------------------------------------------------------------------
    # Step 5: Convert to classification
    # -------------------------------------------------------------------------
    def _assign_classes(self, y):
        """Convert regression targets to class labels."""
        return self.class_assigner(y).float()

    # -------------------------------------------------------------------------
    # Step 6: Scale and pad features
    # -------------------------------------------------------------------------
    def _scale_and_pad_features(self, x):
        """Scale by feature ratio and pad with zeros."""
        num_used = self.h['num_features_used']
        num_total = self.args['num_features']

        # Scale to compensate for zero padding
        if self.h.get('normalize_by_used_features', False):
            x = normalize_by_used_features_f(
                x, num_used, num_total,
                normalize_with_sqrt=self.h.get('normalize_with_sqrt', False)
            )

        # Pad with zeros
        if num_used < num_total:
            padding = torch.zeros(
                (x.shape[0], x.shape[1], num_total - num_used),
                device=self.args['device']
            )
            x = torch.cat([x, padding], dim=-1)

        return x

    # -------------------------------------------------------------------------
    # Step 7: Ensure train/test compatibility
    # -------------------------------------------------------------------------
    def _ensure_class_compatibility(self, x, y):
        """Ensure train and test sets have the same classes."""
        if not self.h.get('check_is_compatible', False):
            return x, y

        eval_pos = self.args.get('single_eval_pos', x.shape[0] // 2)

        for b in range(y.shape[1]):
            for _ in range(10):  # max retries
                train_classes = torch.unique(y[:eval_pos, b], sorted=True)
                test_classes = torch.unique(y[eval_pos:, b], sorted=True)

                is_compatible = (
                    len(train_classes) == len(test_classes) and
                    (train_classes == test_classes).all() and
                    len(train_classes) > 1
                )

                if is_compatible:
                    break

                # Shuffle to try to get compatible classes
                perm = torch.randperm(x.shape[0])
                x[:, b], y[:, b] = x[perm, b], y[perm, b]
            else:
                # Mark as invalid (ignore in loss)
                y[:, b] = -100

        return x, y

    # -------------------------------------------------------------------------
    # Step 8: Normalize labels
    # -------------------------------------------------------------------------
    def _normalize_labels(self, y):
        """Remap labels to consecutive integers (0, 1, 2, ...)."""
        if not self.h.get('normalize_labels', False):
            return y

        for b in range(y.shape[1]):
            valid = y[:, b] != -100
            if self.h.get('normalize_ignore_label_too', False):
                valid[:] = True

            if valid.sum() == 0:
                continue

            # Remap to 0, 1, 2, ...
            unique_labels = y[valid, b].unique()
            y[valid, b] = (y[valid, b] > unique_labels.unsqueeze(1)).sum(axis=0).float()

            # Random rotation of labels
            if self.h.get('rotate_normalized_labels', True) and y[valid, b].numel() > 0:
                num_classes = int(y[valid, b].max().item()) + 1
                shift = torch.randint(0, num_classes, (1,), device=self.args['device'])
                y[valid, b] = (y[valid, b] + shift) % num_classes

        return y

    # -------------------------------------------------------------------------
    # Main forward pass
    # -------------------------------------------------------------------------
    def forward(self, batch_size):
        # Step 1: Generate raw data
        x, y, _ = self._generate_raw_data()

        # Step 2: Add missing values
        x = self._add_missing_values(x)
        # print(f"After missing: {torch.isnan(x).sum().item()}")
        
        # Step 3: Make some features categorical
        x = self._make_categorical_features(x)
        # print(f"After categorical: {torch.isnan(x).sum().item()}")

        # Step 4: Normalize features
        x, y = self._normalize_features(x, y)
        # print(f"After normalize: {torch.isnan(x).sum().item()}")
        
        # Step 5: Convert to classification
        y = self._assign_classes(y)

        # Step 6: Scale and pad features
        x = self._scale_and_pad_features(x)

        # Step 7: Ensure train/test class compatibility
        x, y = self._ensure_class_compatibility(x, y)

        # Step 8: Normalize labels
        y = self._normalize_labels(y)

        return x, y, y


# =============================================================================
# Module-level get_batch function
# =============================================================================

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, device,
              hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    """
    Generate a batch of data using FlexibleCategorical wrapper.

    Args:
        batch_size: Total batch size
        seq_len: Sequence length (number of samples)
        num_features: Total number of features (including padding)
        get_batch: Underlying data generator (e.g., mlp.get_batch)
        device: torch device
        hyperparameters: Configuration dict
        batch_size_per_gp_sample: Sub-batch size for each model
    """
    batch_size_per_gp_sample = batch_size_per_gp_sample or min(32, batch_size)
    num_models = batch_size // batch_size_per_gp_sample

    assert num_models > 0, f'Batch size ({batch_size}) too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    # Resolve seq_len and num_features_used if they are samplers
    if 'seq_len_used' in hyperparameters and callable(hyperparameters['seq_len_used']):
        seq_len = hyperparameters['seq_len_used']()

    if 'num_features_used' in hyperparameters and callable(hyperparameters['num_features_used']):
        hyperparameters['num_features_used'] = hyperparameters['num_features_used']()
    else:
        hyperparameters['num_features_used'] = hyperparameters.get('num_features_used', num_features)

    args = {
        'device': device,
        'seq_len': seq_len,
        'num_features': num_features,
        'batch_size': batch_size_per_gp_sample,
        **kwargs
    }

    # Create models and generate samples
    models = [FlexibleCategorical(get_batch, hyperparameters, args).to(device) for _ in range(num_models)]
    samples = [model(batch_size=batch_size_per_gp_sample) for model in models]

    # Concatenate results
    x, y, y_ = zip(*samples)
    x = torch.cat(x, dim=1).detach()
    y = torch.cat(y, dim=1).detach()
    y_ = torch.cat(y_, dim=1).detach()

    return x, y, y_


DataLoader = get_batch_to_dataloader(get_batch)
