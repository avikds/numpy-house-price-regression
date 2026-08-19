"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.asarray(X, dtype=float).copy()

    # Calculate column-wise means while ignoring NaNs.
    means = np.nanmean(X, axis=0)

    # Replace means of all-NaN columns with 0.0.
    means = np.where(np.isnan(means), 0.0, means)

    # Find NaN entries and replace them with their column mean.
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = means[nan_cols]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    """Compute per-column outlier clip bounds using the IQR rule.

    Args:
        X: (N, F) array-like of numeric features.
        k: IQR multiplier, default 1.5.

    Returns:
        lower: (F,) ndarray of lower bounds.
        upper: (F,) ndarray of upper bounds.
    """
    X = np.asarray(X, dtype=float)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    """Clip every entry of X to per-column lower and upper bounds.

    Args:
        X: (N, F) array-like of numeric features.
        lower: (F,) array-like of lower bounds.
        upper: (F,) array-like of upper bounds.

    Returns:
        (N, F) ndarray with values clipped per column.
    """
    X = np.asarray(X, dtype=float)

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    """Form a derived ratio feature using safe division.

    Args:
        numerator: (N,) array-like numerator values.
        denominator: (N,) array-like denominator values.
        eps: Small value added to the denominator for numerical safety.

    Returns:
        (N,) ndarray containing numerator / (denominator + eps).
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    """Horizontally append one 1-D feature column onto a design matrix.

    Args:
        X: (N, F) array-like design matrix.
        col: (N,) array-like feature column.

    Returns:
        (N, F+1) ndarray with col appended as the last column.
    """
    X = np.asarray(X, dtype=float)
    col = np.asarray(col, dtype=float).reshape(-1, 1)

    return np.hstack((X, col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    """Convert categorical labels into a dense binary one-hot matrix.

    Args:
        labels: (N,) array-like of categorical labels.

    Returns:
        (N, C) float ndarray, where columns correspond to sorted unique labels.
    """
    labels = np.asarray(labels)

    unique_labels, inverse = np.unique(labels, return_inverse=True)

    one_hot = np.zeros((labels.shape[0], unique_labels.shape[0]), dtype=float)
    one_hot[np.arange(labels.shape[0]), inverse] = 1.0

    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    """Compute per-column mean and standard deviation for standardization.

    Args:
        X: (N, F) array-like of numeric features.

    Returns:
        mean: (F,) ndarray of column means.
        std: (F,) ndarray of column standard deviations, with zeros replaced by 1.0.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1.0, std)

    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    """Standardize a numeric feature matrix using fitted column statistics.

    Args:
        X: (N, F) array-like of numeric features.
        mean: (F,) array-like of column means.
        std: (F,) array-like of column standard deviations.

    Returns:
        (N, F) ndarray containing (X - mean) / std.
    """
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    """Prepend a column of ones to a 2-D feature matrix.

    Args:
        X: (N, F) array-like feature matrix.

    Returns:
        (N, F+1) ndarray with a bias column of 1.0 prepended.
    """
    X = np.asarray(X, dtype=float)

    bias = np.ones((X.shape[0], 1), dtype=float)

    return np.hstack((bias, X))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    """Create a reproducibly shuffled permutation of row indices.

    Args:
        n_samples: Number of samples.
        seed: Integer random seed.

    Returns:
        1-D NumPy integer array containing each index exactly once.
    """
    rng = np.random.default_rng(seed)

    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    """Split indices into train, validation, and test subsets.

    Args:
        indices: 1-D array-like of row indices.
        train_ratio: Fraction of samples assigned to training.
        val_ratio: Fraction of samples assigned to validation.

    Returns:
        train_idx, val_idx, test_idx: 1-D integer ndarrays forming a partition.
    """
    indices = np.asarray(indices, dtype=int)

    n_samples = len(indices)

    train_end = int(np.floor(n_samples * train_ratio))
    val_end = train_end + int(np.floor(n_samples * val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    """Select rows of X and y at the given indices.

    Args:
        X: (N, F) array-like feature matrix.
        y: (N,) array-like target vector.
        indices: 1-D array-like row indices.

    Returns:
        X_sub: Feature rows corresponding to indices.
        y_sub: Target values corresponding to indices.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    indices = np.asarray(indices, dtype=int)

    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    """Return the ordinary-least-squares weight vector.

    Args:
        X: (N, D) design matrix including a bias column.
        y: (N,) target vector.

    Returns:
        theta: (D,) OLS weight vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    xtx = X.T @ X
    xty = X.T @ y

    theta = np.linalg.solve(xtx, xty)

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    """Predict continuous targets with a fitted linear model.

    Args:
        X: (N, D) feature matrix.
        theta: (D,) fitted weight vector.

    Returns:
        (N,) ndarray of predicted targets.
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)

    return X @ theta

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

