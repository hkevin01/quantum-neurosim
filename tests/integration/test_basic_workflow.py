"""
Integration test for basic quantum neural network workflow.

This test verifies that the core components work together
for a complete training and prediction workflow.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from qns.data import generate_xor_data
from qns.encoders import AngleEncoder
from qns.models import QuantumClassifier


@pytest.mark.integration
class TestBasicWorkflow:
    """Test complete quantum neural network workflow."""

    def test_end_to_end_xor_training(self):
        """Test complete XOR training workflow."""
        # Generate training data
        X_train, y_train = generate_xor_data(n_samples=20, noise_level=0.1)

        # Create encoder
        encoder = AngleEncoder(n_qubits=2)

        # Create quantum classifier
        classifier = QuantumClassifier(n_qubits=2, n_layers=1)

        # Mock quantum operations for testing
        with patch.object(classifier, '_build_circuit') as mock_build:
            mock_build.return_value = Mock()
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 60, '1': 40}

                # Train the model
                classifier.fit(X_train, y_train, epochs=3)

                # Make predictions
                predictions = classifier.predict(X_train)

                # Verify results
                assert len(predictions) == len(y_train)
                assert all(pred in [0, 1] for pred in predictions)
                assert classifier.is_fitted

    def test_encoder_model_integration(self):
        """Test encoder and model integration."""
        # Create test data
        X = np.array([[0.5, 0.8], [0.3, 0.6]])

        # Test encoder
        encoder = AngleEncoder(n_qubits=2)

        # Should be able to encode data
        circuits = []
        for sample in X:
            circuit = encoder.encode(sample)
            circuits.append(circuit)

        assert len(circuits) == 2
        assert all(circuit is not None for circuit in circuits)

    def test_data_pipeline_integration(self):
        """Test complete data processing pipeline."""
        # Generate different types of data
        xor_X, xor_y = generate_xor_data(n_samples=50)

        # Should be able to process with quantum classifier
        classifier = QuantumClassifier(n_qubits=2)

        with patch.object(classifier, '_build_circuit'):
            with patch.object(classifier, '_execute_circuit') as mock_execute:
                mock_execute.return_value = {'0': 50, '1': 50}

                # Should handle the data without errors
                classifier.fit(xor_X, xor_y, epochs=2)
                predictions = classifier.predict(xor_X)

                assert len(predictions) == len(xor_y)

    def test_error_handling_integration(self):
        """Test error handling across components."""
        classifier = QuantumClassifier(n_qubits=2)

        # Should handle prediction before training
        X_test = np.array([[0.1, 0.2]])
        with pytest.raises(Exception):  # Should raise some error
            classifier.predict(X_test)

    def test_configuration_consistency(self):
        """Test that configurations are consistent across components."""
        n_qubits = 3

        encoder = AngleEncoder(n_qubits=n_qubits)
        classifier = QuantumClassifier(n_qubits=n_qubits)

        # Should have consistent qubit counts
        assert encoder.n_qubits == n_qubits
        assert classifier.config.n_qubits == n_qubits

        # Should be compatible for data with correct dimensions
        X = np.random.rand(10, n_qubits)

        # Encoder should handle the data
        circuit = encoder.encode(X[0])
        assert circuit is not None
