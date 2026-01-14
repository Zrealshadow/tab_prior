"""
Complete Guide: How to Configure flexible_categorical Hyperparameters
"""


def get_base_hyperparameters():
    """Base hyperparameters template with explanations"""

    hyperparameters = {
        # ----------------------------------------------------------------
        # 1. BASE PRIOR PARAMETERS (for MLP or GP)
        # ----------------------------------------------------------------
        # These are passed to the underlying method (mlp.get_batch or fast_gp.get_batch)

        'num_layers': 3,                      # MLP depth
        'prior_mlp_hidden_dim': 64,           # MLP hidden size
        'is_causal': True,                    # Use causal structure (SCM)
        'num_causes': 10,                     # Number of causal variables
        'noise_std': 0.1,                     # Gaussian noise level
        'y_is_effect': True,                  # Target depends on features
        'pre_sample_weights': True,           # Pre-sample MLP weights
        'prior_mlp_dropout_prob': 0.2,        # Dropout probability
        'pre_sample_causes': True,            # Pre-sample causal structure
        'prior_mlp_activations': 'relu',  # Activation function
        'block_wise_dropout': False,
        'prior_mlp_scale_weights_sqrt': True,
        'init_std': 1.0,
        'mix_activations': False,
        'sort_features': False,
        'in_clique': False,
        'random_feature_rotation': False,
        'sampling': 'normal',                 # 'normal', 'uniform', or 'mixed'
        'new_mlp_per_example': False,

        # ----------------------------------------------------------------
        # 2. CLASSIFICATION PARAMETERS
        # ----------------------------------------------------------------
        'num_classes': 3,                     # 0=regression, 2=binary, 3+=multiclass
        'balanced': True,                     # Balance class distribution (binary only)
        'multiclass_type': 'rank',            # 'rank', 'value', or 'multi_node'
        'output_multiclass_ordered_p': 0.5,   # Prob of ordered classes (0.0=unordered)

        # ----------------------------------------------------------------
        # 3. MISSING DATA PARAMETERS
        # ----------------------------------------------------------------
        'nan_prob_no_reason': 0.1,            # Random missing (MCAR) probability
        'nan_prob_a_reason': 0.05,            # Structured missing (MAR) probability
        'nan_prob_unknown_reason': 0.0,       # Unknown mechanism probability
        'nan_prob_unknown_reason_reason_prior': 0.5,  # Mix of above two
        'set_value_to_nan': 0.8,              # 1.0=use NaN, 0.0=use sentinel values

        # ----------------------------------------------------------------
        # 4. CATEGORICAL FEATURES
        # ----------------------------------------------------------------
        'categorical_feature_p': 0.3,         # Probability of categorical features

        # ----------------------------------------------------------------
        # 5. DATA PREPROCESSING
        # ----------------------------------------------------------------
        'normalize_to_ranking': False,        # Use rank transform (True) or z-score (False)

        # ----------------------------------------------------------------
        # 6. FEATURE SELECTION
        # ----------------------------------------------------------------
        'normalize_by_used_features': True,   # Normalize by num_features_used
        'num_features_used': 15,              # Number of informative features
        'seq_len_used': 80,                   # Number of samples used

        # ----------------------------------------------------------------
        # 7. LABEL PROCESSING
        # ----------------------------------------------------------------
        'normalize_labels': True,             # Remap labels to 0,1,2,...
        'check_is_compatible': False,         # Ensure train/test class overlap
        'normalize_ignore_label_too': False,  # Include -100 labels in normalization
        'rotate_normalized_labels': True,     # Randomly rotate class labels
    }

    return hyperparameters


QUICK_REFERENCE = """
QUICK REFERENCE: Key Parameters

+------------------------------------------------------------------+
| CLASSIFICATION                                                   |
+------------------------------------------------------------------+
| num_classes: 0              -> Regression                        |
| num_classes: 2              -> Binary classification             |
| num_classes: 3+             -> Multi-class classification        |
| balanced: True              -> 50/50 split (binary only)         |
| multiclass_type: 'rank'     -> Rank-based classes                |
| multiclass_type: 'value'    -> Fixed threshold classes           |
+------------------------------------------------------------------+
| MISSING DATA                                                     |
+------------------------------------------------------------------+
| nan_prob_no_reason: 0.2     -> 20% random missing (MCAR)         |
| nan_prob_a_reason: 0.1      -> 10% structured missing (MAR)      |
| set_value_to_nan: 1.0       -> Use actual NaN                    |
| set_value_to_nan: 0.0       -> Use sentinel values (-999,0,1,999)|
+------------------------------------------------------------------+
| CATEGORICAL FEATURES                                             |
+------------------------------------------------------------------+
| categorical_feature_p: 0.3  -> 30% features are categorical      |
+------------------------------------------------------------------+
| FEATURE SELECTION                                                |
+------------------------------------------------------------------+
| num_features_used: 15       -> 15 informative features           |
| (if num_features=20)        -> 5 features are zero-padded        |
| normalize_by_used_features  -> Scale by informative ratio        |
+------------------------------------------------------------------+
| PREPROCESSING                                                    |
+------------------------------------------------------------------+
| normalize_to_ranking: False -> Z-score normalization             |
| normalize_to_ranking: True  -> Rank transformation               |
| normalize_labels: True      -> Remap labels to 0,1,2,...         |
| rotate_normalized_labels    -> Randomly shift labels             |
+------------------------------------------------------------------+
"""
