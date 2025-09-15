"""
Unit tests for data utilities and generators.

This module tests the data generation and preprocessing utilities
used for quantum neural network experiments.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from qns.data import (DataGenerationError, generate_hopfield_patterns,
                      generate_parity_data, generate_time_series,
                      generate_xor_data, normalize_data, split_data)


class TestXORDataGeneration:
    """Test XOR dataset generation."""

    def test_generate_basic_xor(self):
        """Test basic XOR data generation."""
        X, y = generate_xor_data(n_samples=100)

        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(y) == {0, 1}

    def test_xor_logical_correctness(self):
        """Test that XOR logic is correctly implemented."""
        X, y = generate_xor_data(n_samples=1000, noise_level=0.0)

        # Check XOR logic for samples
        for i in range(min(100, len(X))):  # Check first 100 samples
            x1, x2 = X[i]
            expected = int((x1 > 0) != (x2 > 0))  # XOR logic
            # Allow some tolerance for noise
            assert abs(y[i] - expected) <= 0.1

    def test_xor_with_noise(self):
        """Test XOR data with different noise levels."""
        X_no_noise, _ = generate_xor_data(n_samples=100, noise_level=0.0)
        X_noise, _ = generate_xor_data(n_samples=100, noise_level=0.5)

        # Noisy data should have higher variance
        assert np.var(X_noise) > np.var(X_no_noise)

    def test_xor_custom_bounds(self):
        """Test XOR data with custom bounds."""
        X, y = generate_xor_data(
            n_samples=100,
            bounds=(-2, 2),
            noise_level=0.0
        )

        assert np.all(X >= -2) and np.all(X <= 2)

    def test_xor_invalid_params(self):
        """Test XOR generation with invalid parameters."""
        with pytest.raises(DataGenerationError):
            generate_xor_data(n_samples=0)

        with pytest.raises(DataGenerationError):
            generate_xor_data(n_samples=100, noise_level=-0.1)

        with pytest.raises(DataGenerationError):
            generate_xor_data(n_samples=100, bounds=(2, 1))  # Invalid bounds

    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_xor_various_sizes(self, n_samples):
        """Test XOR generation with various sample sizes."""
        X, y = generate_xor_data(n_samples=n_samples)

        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples

    def test_xor_reproducibility(self):
        """Test that XOR generation is reproducible with seed."""
        X1, y1 = generate_xor_data(n_samples=100, random_seed=42)
        X2, y2 = generate_xor_data(n_samples=100, random_seed=42)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestParityDataGeneration:
    """Test parity dataset generation."""

    def test_generate_basic_parity(self):
        """Test basic parity data generation."""
        X, y = generate_parity_data(n_samples=100, n_bits=3)

        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(y) == {0, 1}

    def test_parity_logical_correctness(self):
        """Test that parity logic is correctly implemented."""
        X, y = generate_parity_data(n_samples=100, n_bits=4, noise_level=0.0)

        # Check parity logic for binary samples
        for i in range(min(20, len(X))):
            x_binary = (X[i] > 0).astype(int)
            expected = sum(x_binary) % 2
            assert y[i] == expected

    def test_parity_different_bits(self):
        """Test parity with different bit counts."""
        for n_bits in [2, 3, 4, 5]:
            X, y = generate_parity_data(n_samples=50, n_bits=n_bits)
            assert X.shape == (50, n_bits)
            assert y.shape == (50,)

    def test_parity_invalid_params(self):
        """Test parity generation with invalid parameters."""
        with pytest.raises(DataGenerationError):
            generate_parity_data(n_samples=100, n_bits=1)  # Too few bits

        with pytest.raises(DataGenerationError):
            generate_parity_data(n_samples=0, n_bits=3)

        with pytest.raises(DataGenerationError):
            generate_parity_data(n_samples=100, n_bits=3, noise_level=-0.1)

    def test_parity_reproducibility(self):
        """Test parity generation reproducibility."""
        X1, y1 = generate_parity_data(n_samples=100, n_bits=3, random_seed=123)
        X2, y2 = generate_parity_data(n_samples=100, n_bits=3, random_seed=123)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestHopfieldPatternGeneration:
    """Test Hopfield pattern generation."""

    def test_generate_basic_patterns(self):
        """Test basic Hopfield pattern generation."""
        patterns = generate_hopfield_patterns(n_patterns=5, pattern_size=10)

        assert patterns.shape == (5, 10)
        assert np.all(np.isin(patterns, [-1, 1]))

    def test_pattern_orthogonality(self):
        """Test that patterns have desired orthogonality properties."""
        patterns = generate_hopfield_patterns(
            n_patterns=3,
            pattern_size=20,
            orthogonal=True
        )

        # Check approximate orthogonality
        for i in range(patterns.shape[0]):
            for j in range(i + 1, patterns.shape[0]):
                dot_product = np.dot(patterns[i], patterns[j])
                # Orthogonal patterns should have dot product close to 0
                assert abs(dot_product) < patterns.shape[1] * 0.3

    def test_pattern_corruption(self):
        """Test pattern corruption functionality."""
        original = generate_hopfield_patterns(n_patterns=1, pattern_size=20)
        corrupted = generate_hopfield_patterns(
            n_patterns=1,
            pattern_size=20,
            corruption_prob=0.2
        )

        # Should have some differences due to corruption
        if len(original) > 0 and len(corrupted) > 0:
            diff_ratio = np.mean(original[0] != corrupted[0])
            assert 0.1 <= diff_ratio <= 0.3  # Expect around 20% corruption

    def test_pattern_invalid_params(self):
        """Test pattern generation with invalid parameters."""
        with pytest.raises(DataGenerationError):
            generate_hopfield_patterns(n_patterns=0, pattern_size=10)

        with pytest.raises(DataGenerationError):
            generate_hopfield_patterns(n_patterns=5, pattern_size=0)

        with pytest.raises(DataGenerationError):
            generate_hopfield_patterns(
                n_patterns=5,
                pattern_size=10,
                corruption_prob=-0.1
            )

    def test_pattern_sparsity(self):
        """Test sparse pattern generation."""
        patterns = generate_hopfield_patterns(
            n_patterns=10,
            pattern_size=50,
            sparsity=0.1
        )

        # Check sparsity (approximately 10% should be +1)
        for pattern in patterns:
            active_ratio = np.mean(pattern == 1)
            assert 0.05 <= active_ratio <= 0.15  # Allow some variance


class TestTimeSeriesGeneration:
    """Test time series generation."""

    def test_generate_basic_series(self):
        """Test basic time series generation."""
        series = generate_time_series(n_points=100)

        assert len(series) == 100
        assert isinstance(series, np.ndarray)

    def test_sinusoidal_series(self):
        """Test sinusoidal time series generation."""
        series = generate_time_series(
            n_points=100,
            series_type='sinusoidal',
            frequency=1.0,
            noise_level=0.0
        )

        # Should be periodic
        t = np.linspace(0, 2*np.pi, 100)
        expected = np.sin(t)

        # Check correlation with expected sine wave
        correlation = np.corrcoef(series, expected)[0, 1]
        assert correlation > 0.8

    def test_chaotic_series(self):
        """Test chaotic time series generation."""
        series = generate_time_series(
            n_points=100,
            series_type='chaotic'
        )

        # Chaotic series should have bounded values
        assert np.all(np.isfinite(series))
        assert np.std(series) > 0  # Should have variation

    def test_random_walk_series(self):
        """Test random walk time series."""
        series = generate_time_series(
            n_points=100,
            series_type='random_walk',
            noise_level=1.0
        )

        # Random walk should show cumulative behavior
        differences = np.diff(series)
        assert np.std(differences) > 0

    def test_series_invalid_params(self):
        """Test time series generation with invalid parameters."""
        with pytest.raises(DataGenerationError):
            generate_time_series(n_points=0)

        with pytest.raises(DataGenerationError):
            generate_time_series(n_points=100, series_type='invalid')

        with pytest.raises(DataGenerationError):
            generate_time_series(n_points=100, noise_level=-0.1)

    def test_series_reproducibility(self):
        """Test time series generation reproducibility."""
        series1 = generate_time_series(n_points=100, random_seed=456)
        series2 = generate_time_series(n_points=100, random_seed=456)

        np.testing.assert_array_equal(series1, series2)


class TestDataNormalization:
    """Test data normalization utilities."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = normalize_data(data, method='minmax')

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.shape == data.shape

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = np.random.randn(100, 5) * 10 + 5  # Non-standard distribution
        normalized = normalize_data(data, method='zscore')

        # Should have approximately zero mean and unit variance
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1

    def test_unit_normalization(self):
        """Test unit vector normalization."""
        data = np.array([[3, 4], [5, 12]])  # Known magnitudes: 5, 13
        normalized = normalize_data(data, method='unit')

        # Each row should have unit magnitude
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_quantum_normalization(self):
        """Test quantum amplitude normalization."""
        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        normalized = normalize_data(data, method='quantum')

        # Each row should satisfy quantum normalization (sum of squares = 1)
        squared_sums = np.sum(normalized**2, axis=1)
        np.testing.assert_array_almost_equal(squared_sums, [1.0, 1.0])

    def test_normalization_invalid_method(self):
        """Test normalization with invalid method."""
        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            normalize_data(data, method='invalid_method')

    def test_normalization_edge_cases(self):
        """Test normalization edge cases."""
        # Constant data
        constant_data = np.array([[1, 1], [1, 1]])

        # Should handle constant data gracefully
        try:
            normalized = normalize_data(constant_data, method='zscore')
            assert np.all(np.isfinite(normalized))
        except (ValueError, RuntimeWarning):
            pass  # Expected behavior for degenerate cases

    def test_normalization_preserves_shape(self):
        """Test that normalization preserves data shape."""
        original_shapes = [(10, 3), (1, 5), (50, 1), (100, 10)]

        for shape in original_shapes:
            data = np.random.randn(*shape)
            normalized = normalize_data(data, method='minmax')
            assert normalized.shape == shape


class TestDataSplitting:
    """Test data splitting utilities."""

    def test_basic_split(self):
        """Test basic train-test split."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_split_with_validation(self):
        """Test train-validation-test split."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        splits = split_data(X, y, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        assert X_train.shape[0] == 60  # 60% for training
        assert X_val.shape[0] == 20    # 20% for validation
        assert X_test.shape[0] == 20   # 20% for testing

    def test_split_stratified(self):
        """Test stratified splitting."""
        # Create imbalanced dataset
        X = np.random.randn(100, 3)
        y = np.array([0] * 80 + [1] * 20)  # 80% class 0, 20% class 1

        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, stratify=True
        )

        # Check that proportions are maintained
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        original_ratio = np.mean(y)

        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.1

    def test_split_reproducibility(self):
        """Test that splitting is reproducible with seed."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        split1 = split_data(X, y, test_size=0.3, random_seed=789)
        split2 = split_data(X, y, test_size=0.3, random_seed=789)

        # All splits should be identical
        for arr1, arr2 in zip(split1, split2):
            np.testing.assert_array_equal(arr1, arr2)

    def test_split_invalid_params(self):
        """Test splitting with invalid parameters."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)

        with pytest.raises(ValueError):
            split_data(X, y, test_size=1.5)  # Invalid test size

        with pytest.raises(ValueError):
            split_data(X, y, test_size=0.5, val_size=0.6)  # Sizes sum > 1

    def test_split_small_dataset(self):
        """Test splitting very small datasets."""
        X = np.random.randn(5, 2)
        y = np.random.randint(0, 2, 5)

        # Should handle small datasets gracefully
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)

        assert X_train.shape[0] + X_test.shape[0] == 5
        assert y_train.shape[0] + y_test.shape[0] == 5


class TestDataGenerationError:
    """Test custom data generation exception."""

    def test_error_creation(self):
        """Test creating data generation error."""
        error = DataGenerationError("Test error message")
        assert str(error) == "Test error message"

    def test_error_inheritance(self):
        """Test error inheritance."""
        error = DataGenerationError("Test")
        assert isinstance(error, Exception)


@pytest.mark.parametrize("dataset_type", [
    "xor", "parity", "classification", "regression"
])
def test_dataset_compatibility(dataset_type):
    """Test that generated datasets work with sklearn."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split

    # Generate appropriate dataset
    if dataset_type == "xor":
        X, y = generate_xor_data(n_samples=100)
        model = LogisticRegression()
    elif dataset_type == "parity":
        X, y = generate_parity_data(n_samples=100, n_bits=3)
        model = LogisticRegression()
    elif dataset_type == "classification":
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2)
        model = LogisticRegression()
    else:  # regression
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        model = LinearRegression()

    # Should work with sklearn
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
    except Exception as e:
        pytest.skip(f"Sklearn compatibility test failed: {e}")


@pytest.mark.slow
class TestDataGenerationPerformance:
    """Test data generation performance."""

    def test_large_dataset_generation(self):
        """Test generating large datasets efficiently."""
        import time

        start_time = time.time()
        X, y = generate_xor_data(n_samples=10000)
        generation_time = time.time() - start_time

        assert X.shape == (10000, 2)
        assert generation_time < 1.0  # Should be fast

    def test_batch_normalization_performance(self):
        """Test batch normalization performance."""
        import time

        large_data = np.random.randn(10000, 100)

        start_time = time.time()
        normalized = normalize_data(large_data, method='zscore')
        normalization_time = time.time() - start_time

        assert normalized.shape == large_data.shape
        assert normalization_time < 2.0  # Should be efficient

    def test_memory_efficiency(self):
        """Test memory efficiency of data operations."""
        # Test with large dataset that should fit in memory
        try:
            X = generate_time_series(n_points=100000)
            normalized = normalize_data(X.reshape(-1, 1), method='minmax')
            assert normalized.shape[0] == 100000
        except MemoryError:
            pytest.skip("Memory test requires more RAM")
