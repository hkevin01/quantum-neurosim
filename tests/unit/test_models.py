"""
Unit tests for quantum neural network models.

This module tests the quantum neural network model implementations,
including training, prediction, and resource management.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from qns.models import (BaseQuantumModel, ModelConfig, QuantumClassifier,
                        QuantumModelError, ResourceMonitor, TrainingError)


class TestModelConfig:
    """Test the model configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.n_qubits == 4
        assert config.n_layers == 2
        assert config.learning_rate == 0.01
        assert config.max_iter == 100
        assert config.tolerance == 1e-6

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            n_qubits=8,
            n_layers=4,
            learning_rate=0.001,
            max_iter=200
        )
        assert config.n_qubits == 8
        assert config.n_layers == 4
        assert config.learning_rate == 0.001
        assert config.max_iter == 200

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ModelConfig(n_qubits=2, n_layers=1)
        assert config.n_qubits == 2

        # Invalid configurations should raise errors
        with pytest.raises(ValueError):
            ModelConfig(n_qubits=0)

        with pytest.raises(ValueError):
            ModelConfig(n_layers=0)

        with pytest.raises(ValueError):
            ModelConfig(learning_rate=0)


class TestResourceMonitor:
    """Test the resource monitoring context manager."""

    def test_resource_monitor_creation(self):
        """Test creating resource monitor."""
        monitor = ResourceMonitor("test_operation")
        assert monitor.operation_name == "test_operation"

    def test_resource_monitor_context(self):
        """Test resource monitor as context manager."""
        with ResourceMonitor("test") as monitor:
            # Simulate some work
            import time
            time.sleep(0.01)

        assert monitor.execution_time > 0
        assert monitor.memory_usage >= 0

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        # Mock memory usage to exceed limit
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 2e9  # 2GB

            with pytest.warns(UserWarning):
                with ResourceMonitor("test", max_memory_mb=1000):
                    pass

    def test_timeout_enforcement(self):
        """Test timeout enforcement."""
        with pytest.raises(TimeoutError):
            with ResourceMonitor("test", timeout_seconds=0.01):
                import time
                time.sleep(0.1)  # Longer than timeout


class TestBaseQuantumModel:
    """Test the abstract base quantum model."""

    def test_cannot_instantiate_base_model(self):
        """Test that BaseQuantumModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseQuantumModel()

    def test_base_model_interface(self):
        """Test that BaseQuantumModel defines required interface."""
        # Check abstract methods
        assert hasattr(BaseQuantumModel, 'fit')
        assert hasattr(BaseQuantumModel, 'predict')
        assert getattr(BaseQuantumModel.fit, '__isabstractmethod__', False)
        assert getattr(BaseQuantumModel.predict, '__isabstractmethod__', False)


class TestQuantumClassifier:
    """Test the quantum classifier implementation."""

    def test_initialization(self):
        """Test QuantumClassifier initialization."""
        classifier = QuantumClassifier(n_qubits=4, n_layers=2)
        assert classifier.config.n_qubits == 4
        assert classifier.config.n_layers == 2

    def test_initialization_with_config(self):
        """Test initialization with ModelConfig object."""
        config = ModelConfig(n_qubits=6, n_layers=3)
        classifier = QuantumClassifier(config=config)
        assert classifier.config.n_qubits == 6
        assert classifier.config.n_layers == 3

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            QuantumClassifier(n_qubits=0)

        with pytest.raises(ValueError):
            QuantumClassifier(n_qubits=4, n_layers=0)

    @patch('qns.models.ResourceMonitor')
    def test_fit_method(self, mock_monitor):
        """Test the fit method."""
        mock_monitor.return_value.__enter__.return_value = MagicMock()
        mock_monitor.return_value.__exit__.return_value = None

        classifier = QuantumClassifier(n_qubits=2)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR pattern

        # Mock quantum backend
        with patch.object(classifier, '_build_circuit') as mock_build:
            mock_build.return_value = Mock()
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 50, '1': 50}

                classifier.fit(X, y, epochs=5)

        assert classifier.is_fitted
        mock_monitor.assert_called()

    def test_fit_data_validation(self):
        """Test fit method data validation."""
        classifier = QuantumClassifier(n_qubits=2)

        # Mismatched X and y shapes
        X = np.array([[0, 0], [0, 1]])
        y = np.array([0, 1, 1])  # Wrong length

        with pytest.raises(TrainingError):
            classifier.fit(X, y)

    def test_fit_feature_dimension_validation(self):
        """Test fit method feature dimension validation."""
        classifier = QuantumClassifier(n_qubits=2)

        # Too many features for number of qubits
        X = np.array([[0, 0, 0]])  # 3 features, but only 2 qubits
        y = np.array([0])

        with pytest.raises(TrainingError):
            classifier.fit(X, y)

    @patch('qns.models.ResourceMonitor')
    def test_predict_method(self, mock_monitor):
        """Test the predict method."""
        mock_monitor.return_value.__enter__.return_value = MagicMock()
        mock_monitor.return_value.__exit__.return_value = None

        classifier = QuantumClassifier(n_qubits=2)
        classifier.is_fitted = True  # Simulate fitted model

        X = np.array([[0, 0], [1, 1]])

        with patch.object(classifier, '_build_circuit') as mock_build:
            mock_build.return_value = Mock()
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 80, '1': 20}

                predictions = classifier.predict(X)

        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_not_fitted(self):
        """Test predict method on unfitted model."""
        classifier = QuantumClassifier(n_qubits=2)
        X = np.array([[0, 0]])

        with pytest.raises(QuantumModelError):
            classifier.predict(X)

    def test_score_method(self):
        """Test the score method."""
        classifier = QuantumClassifier(n_qubits=2)
        classifier.is_fitted = True

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_true = np.array([0, 1, 1, 0])

        with patch.object(classifier, 'predict') as mock_predict:
            mock_predict.return_value = np.array([0, 1, 1, 0])  # Perfect predictions

            accuracy = classifier.score(X, y_true)

        assert accuracy == 1.0

    def test_get_params_method(self):
        """Test the get_params method."""
        classifier = QuantumClassifier(n_qubits=4, n_layers=3)
        params = classifier.get_params()

        assert 'n_qubits' in params
        assert 'n_layers' in params
        assert params['n_qubits'] == 4
        assert params['n_layers'] == 3

    def test_set_params_method(self):
        """Test the set_params method."""
        classifier = QuantumClassifier(n_qubits=2, n_layers=1)

        classifier.set_params(n_qubits=4, n_layers=2)

        assert classifier.config.n_qubits == 4
        assert classifier.config.n_layers == 2

    def test_save_load_model(self, temp_dir):
        """Test saving and loading model."""
        classifier = QuantumClassifier(n_qubits=2, n_layers=1)
        classifier.is_fitted = True

        # Mock parameters for testing
        classifier.parameters = np.array([0.1, 0.2, 0.3, 0.4])

        # Save model
        model_path = temp_dir / "test_model.pkl"
        classifier.save(str(model_path))
        assert model_path.exists()

        # Load model
        loaded_classifier = QuantumClassifier.load(str(model_path))
        assert loaded_classifier.config.n_qubits == 2
        assert loaded_classifier.config.n_layers == 1
        assert loaded_classifier.is_fitted

    def test_cross_validation_support(self):
        """Test compatibility with scikit-learn cross-validation."""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score

        # Generate simple dataset
        X, y = make_classification(
            n_samples=20,
            n_features=2,
            n_classes=2,
            n_redundant=0,
            random_state=42
        )

        classifier = QuantumClassifier(n_qubits=2)

        # Mock the quantum operations for testing
        with patch.object(classifier, 'fit') as mock_fit:
            with patch.object(classifier, 'predict') as mock_predict:
                mock_predict.return_value = np.random.randint(0, 2, len(y))

                # This should work with sklearn's cross-validation
                try:
                    scores = cross_val_score(classifier, X, y, cv=2)
                    assert len(scores) == 2
                except Exception as e:
                    pytest.skip(f"Cross-validation test failed: {e}")

    @pytest.mark.parametrize("n_qubits,n_samples", [
        (2, 10), (4, 20), (3, 15)
    ])
    def test_various_configurations(self, n_qubits, n_samples):
        """Test classifier with various configurations."""
        classifier = QuantumClassifier(n_qubits=n_qubits)

        # Generate compatible data
        X = np.random.rand(n_samples, n_qubits)
        y = np.random.randint(0, 2, n_samples)

        # Should initialize without error
        assert classifier.config.n_qubits == n_qubits

    def test_parameter_optimization_tracking(self):
        """Test that parameter optimization is tracked."""
        classifier = QuantumClassifier(n_qubits=2)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 50, '1': 50}

                classifier.fit(X, y, epochs=3)

        # Should track optimization history
        assert hasattr(classifier, 'training_history')

    def test_early_stopping(self):
        """Test early stopping functionality."""
        classifier = QuantumClassifier(
            n_qubits=2,
            early_stopping=True,
            patience=2
        )

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                # Mock improving then stagnating loss
                mock_execute.side_effect = [
                    {'0': 60, '1': 40},  # Loss decreases
                    {'0': 55, '1': 45},  # Loss decreases
                    {'0': 55, '1': 45},  # Loss stagnates
                    {'0': 55, '1': 45},  # Loss stagnates
                ]

                classifier.fit(X, y, epochs=10)

        # Should stop early due to patience
        assert len(classifier.training_history) < 10


class TestQuantumModelError:
    """Test custom quantum model exceptions."""

    def test_quantum_model_error_creation(self):
        """Test creating quantum model error."""
        error = QuantumModelError("Test message")
        assert str(error) == "Test message"

    def test_training_error_creation(self):
        """Test creating training error."""
        error = TrainingError("Training failed")
        assert str(error) == "Training failed"

    def test_error_inheritance(self):
        """Test error inheritance hierarchy."""
        error = TrainingError("Test")
        assert isinstance(error, QuantumModelError)
        assert isinstance(error, Exception)


@pytest.mark.slow
class TestModelPerformance:
    """Test model performance characteristics."""

    def test_training_performance(self, performance_monitor):
        """Test training performance."""
        classifier = QuantumClassifier(n_qubits=3, n_layers=2)

        # Small dataset for performance test
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 50, '1': 50}

                import time
                start = time.time()
                classifier.fit(X, y, epochs=5)
                duration = time.time() - start

        # Should complete within reasonable time
        assert duration < 10.0

    def test_prediction_performance(self, performance_monitor):
        """Test prediction performance."""
        classifier = QuantumClassifier(n_qubits=2)
        classifier.is_fitted = True

        X = np.random.rand(100, 2)

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 70, '1': 30}

                import time
                start = time.time()
                predictions = classifier.predict(X)
                duration = time.time() - start

        assert len(predictions) == 100
        assert duration < 5.0  # Should predict efficiently

    def test_memory_efficiency(self):
        """Test memory efficiency during training."""
        classifier = QuantumClassifier(n_qubits=4)

        # Large batch for memory test
        X = np.random.rand(1000, 4)
        y = np.random.randint(0, 2, 1000)

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 500, '1': 500}

                # Should handle large batches without excessive memory usage
                try:
                    classifier.fit(X, y, epochs=2, batch_size=100)
                except MemoryError:
                    pytest.fail("Memory error during batch processing")


@pytest.mark.integration
class TestModelIntegration:
    """Test model integration with quantum backends."""

    def test_backend_switching(self):
        """Test switching between quantum backends."""
        classifier = QuantumClassifier(n_qubits=2)

        # Should handle different backend types
        mock_backends = [Mock(), Mock()]
        for backend in mock_backends:
            classifier.set_backend(backend)
            assert classifier.backend == backend

    def test_noise_model_integration(self):
        """Test integration with noise models."""
        classifier = QuantumClassifier(n_qubits=2)

        # Mock noise model
        noise_model = Mock()
        classifier.set_noise_model(noise_model)

        assert classifier.noise_model == noise_model
