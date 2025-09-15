"""Data utilities and sample dataset generators for quantum neural networks.

This module provides functions for generating synthetic datasets suitable
for quantum machine learning experiments, data preprocessing utilities,
and benchmark datasets commonly used in quantum computing research.

Functions:
    generate_xor_data: Create XOR classification dataset
    generate_parity_data: Create parity function dataset
    generate_hopfield_patterns: Create patterns for associative memory
    load_quantum_dataset: Load pre-defined quantum datasets
    normalize_data: Normalize data for quantum encoding
    add_noise: Add various types of noise to data

Example:
    >>> from qns.data import generate_xor_data, normalize_data
    >>>
    >>> # Generate XOR dataset
    >>> X, y = generate_xor_data(n_samples=100, noise=0.1)
    >>> X_norm = normalize_data(X, method='minmax')
    >>>
    >>> print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

# Optional imports
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_xor_data(
    n_samples: int = 100,
    noise: float = 0.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR classification dataset.

    Creates a 2D XOR problem that is not linearly separable,
    making it a good test case for quantum neural networks.

    Args:
        n_samples: Number of samples to generate
        noise: Gaussian noise level (0.0 = no noise)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels) arrays

    Example:
        >>> X, y = generate_xor_data(n_samples=200, noise=0.1)
        >>> print(f"Features shape: {X.shape}, Labels: {np.unique(y)}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate balanced samples
    n_per_class = n_samples // 4

    # XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    X = []
    y = []

    # Class 0: (0,0) and (1,1)
    for _ in range(n_per_class):
        X.append([0.0, 0.0])
        y.append(0)
        X.append([1.0, 1.0])
        y.append(0)

    # Class 1: (0,1) and (1,0)
    for _ in range(n_per_class):
        X.append([0.0, 1.0])
        y.append(1)
        X.append([1.0, 0.0])
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Add noise if requested
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    # Shuffle the data
    indices = np.random.permutation(len(X))

    return X[indices], y[indices]


def generate_parity_data(
    n_features: int = 4,
    n_samples: int = 100,
    noise: float = 0.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n-bit parity classification dataset.

    Creates a dataset where the label is the XOR of all feature bits.
    This is a non-linear problem that scales exponentially with features.

    Args:
        n_features: Number of binary features
        n_samples: Number of samples to generate
        noise: Noise level for continuous features
        random_state: Random seed

    Returns:
        Tuple of (features, labels) arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random binary features
    X = np.random.randint(0, 2, size=(n_samples, n_features))

    # Compute parity (XOR of all bits)
    y = np.sum(X, axis=1) % 2

    # Convert to float and add noise
    X = X.astype(float)
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    return X, y


def generate_hopfield_patterns(
    n_patterns: int = 3,
    pattern_size: int = 8,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate random binary patterns for Hopfield network.

    Creates orthogonal or semi-orthogonal binary patterns suitable
    for associative memory experiments.

    Args:
        n_patterns: Number of patterns to generate
        pattern_size: Size of each pattern (number of bits)
        random_state: Random seed

    Returns:
        Array of binary patterns (n_patterns, pattern_size)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random binary patterns
    patterns = np.random.randint(0, 2, size=(n_patterns, pattern_size))

    # Convert 0s to -1s for Hopfield network
    patterns = 2 * patterns - 1

    return patterns


def corrupt_patterns(
    patterns: np.ndarray,
    corruption_rate: float = 0.2,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Add corruption to binary patterns.

    Randomly flips bits in patterns to test associative memory recall.

    Args:
        patterns: Original patterns array
        corruption_rate: Fraction of bits to flip (0.0 to 1.0)
        random_state: Random seed

    Returns:
        Corrupted patterns array
    """
    if random_state is not None:
        np.random.seed(random_state)

    corrupted = patterns.copy()

    for i in range(len(patterns)):
        n_corrupt = int(corruption_rate * patterns.shape[1])
        corrupt_indices = np.random.choice(
            patterns.shape[1], n_corrupt, replace=False
        )
        corrupted[i, corrupt_indices] *= -1  # Flip bits

    return corrupted


def generate_time_series(
    n_samples: int = 100,
    sequence_length: int = 10,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series data.

    Creates sequences with temporal patterns for testing
    quantum recurrent networks or temporal encodings.

    Args:
        n_samples: Number of sequences
        sequence_length: Length of each sequence
        n_features: Number of features per timestep
        noise: Noise level
        random_state: Random seed

    Returns:
        Tuple of (sequences, targets) arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate sinusoidal sequences with different frequencies
    t = np.linspace(0, 4*np.pi, sequence_length)

    sequences = []
    targets = []

    for i in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)

        # Generate sequence
        seq = []
        for f in range(n_features):
            signal = np.sin(freq * t + phase + f * np.pi/4)
            if noise > 0:
                signal += np.random.normal(0, noise, len(signal))
            seq.append(signal)

        sequences.append(np.array(seq).T)

        # Target is the next value (prediction task)
        next_val = np.sin(freq * (t[-1] + t[1] - t[0]) + phase)
        targets.append(next_val)

    return np.array(sequences), np.array(targets)


def normalize_data(
    X: np.ndarray,
    method: str = 'minmax',
    feature_range: Tuple[float, float] = (-1, 1)
) -> np.ndarray:
    """Normalize data for quantum encoding.

    Args:
        X: Input data array
        method: Normalization method ('minmax', 'standard', 'unit')
        feature_range: Target range for minmax scaling

    Returns:
        Normalized data array
    """
    if method == 'minmax':
        # Scale to specified range
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min

        # Handle constant features
        X_range = np.where(X_range == 0, 1, X_range)

        # Scale to [0, 1] then to feature_range
        X_scaled = (X - X_min) / X_range
        X_scaled = X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

        return X_scaled

    elif method == 'standard':
        # Standardize to zero mean and unit variance
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)

        # Handle zero variance features
        X_std = np.where(X_std == 0, 1, X_std)

        return (X - X_mean) / X_std

    elif method == 'unit':
        # Scale to unit norm
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)

        return X / norms

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def add_noise(
    X: np.ndarray,
    noise_type: str = 'gaussian',
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Add various types of noise to data.

    Args:
        X: Input data
        noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
        noise_level: Noise intensity
        random_state: Random seed

    Returns:
        Noisy data array
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_noisy = X.copy()

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy += noise

    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, X.shape)
        X_noisy += noise

    elif noise_type == 'salt_pepper':
        # Random bit flips for binary data
        mask = np.random.random(X.shape) < noise_level
        X_noisy[mask] = 1 - X_noisy[mask]  # Flip bits

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return X_noisy


def load_quantum_dataset(
    name: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-defined quantum machine learning datasets.

    Args:
        name: Dataset name ('xor', 'parity', 'hopfield', 'timeseries')
        **kwargs: Dataset-specific parameters

    Returns:
        Tuple of (features, labels) arrays
    """
    if name.lower() == 'xor':
        return generate_xor_data(**kwargs)
    elif name.lower() == 'parity':
        return generate_parity_data(**kwargs)
    elif name.lower() == 'timeseries':
        return generate_time_series(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# Quantum-specific data utilities
def encode_classical_data(
    X: np.ndarray,
    encoding: str = 'angle',
    n_qubits: Optional[int] = None
) -> np.ndarray:
    """Prepare classical data for quantum encoding.

    Args:
        X: Classical data array
        encoding: Encoding type ('angle', 'amplitude', 'basis')
        n_qubits: Number of qubits (inferred if None)

    Returns:
        Processed data ready for quantum encoding
    """
    if n_qubits is None:
        if encoding == 'amplitude':
            n_qubits = int(np.ceil(np.log2(X.shape[1])))
        else:
            n_qubits = X.shape[1]

    X_processed = X.copy()

    if encoding == 'angle':
        # Normalize to [-1, 1] for angle encoding
        X_processed = normalize_data(X_processed, method='minmax',
                                   feature_range=(-1, 1))

        # Pad or truncate to match qubit count
        if X_processed.shape[1] < n_qubits:
            padding = np.zeros((X_processed.shape[0],
                              n_qubits - X_processed.shape[1]))
            X_processed = np.hstack([X_processed, padding])
        elif X_processed.shape[1] > n_qubits:
            X_processed = X_processed[:, :n_qubits]

    elif encoding == 'amplitude':
        # Normalize to unit norm for amplitude encoding
        X_processed = normalize_data(X_processed, method='unit')

        # Pad to power of 2
        target_size = 2 ** n_qubits
        if X_processed.shape[1] < target_size:
            padding = np.zeros((X_processed.shape[0],
                              target_size - X_processed.shape[1]))
            X_processed = np.hstack([X_processed, padding])
        elif X_processed.shape[1] > target_size:
            X_processed = X_processed[:, :target_size]

    elif encoding == 'basis':
        # Binarize for basis encoding
        X_processed = (X_processed > np.median(X_processed, axis=0)).astype(float)

        # Truncate to qubit count
        if X_processed.shape[1] > n_qubits:
            X_processed = X_processed[:, :n_qubits]

    return X_processed


def benchmark_datasets():
    """Generate a collection of benchmark datasets for testing."""
    datasets = {}

    # Small datasets for quick testing
    datasets['xor_small'] = generate_xor_data(n_samples=40, noise=0.0)
    datasets['parity_3bit'] = generate_parity_data(n_features=3, n_samples=64)
    datasets['parity_4bit'] = generate_parity_data(n_features=4, n_samples=128)

    # Medium datasets for validation
    datasets['xor_noisy'] = generate_xor_data(n_samples=200, noise=0.1)
    datasets['parity_5bit'] = generate_parity_data(n_features=5, n_samples=256)

    # Hopfield patterns
    datasets['hopfield_patterns'] = (
        generate_hopfield_patterns(n_patterns=4, pattern_size=8),
        None  # No labels for unsupervised task
    )

    return datasets


if __name__ == "__main__":
    # Demo of data generation functions
    print("Quantum NeuroSim Data Utilities Demo")
    print("=" * 40)

    # XOR dataset
    X_xor, y_xor = generate_xor_data(n_samples=100, noise=0.05)
    print(f"XOR dataset: {X_xor.shape}, classes: {np.unique(y_xor)}")

    # Parity dataset
    X_parity, y_parity = generate_parity_data(n_features=4, n_samples=100)
    print(f"Parity dataset: {X_parity.shape}, classes: {np.unique(y_parity)}")

    # Hopfield patterns
    patterns = generate_hopfield_patterns(n_patterns=3, pattern_size=6)
    corrupted = corrupt_patterns(patterns, corruption_rate=0.3)
    print(f"Hopfield patterns: {patterns.shape}")

    # Normalization demo
    X_norm = normalize_data(X_xor, method='minmax', feature_range=(-1, 1))
    print(f"Normalized XOR range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
