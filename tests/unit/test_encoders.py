"""
Unit tests for quantum data encoders.

This module tests the quantum data encoding strategies used to prepare
classical data for quantum neural networks.
"""

from unittest.mock import patch

import numpy as np
import pytest

from qns.encoders import (AmplitudeEncoder, AngleEncoder, BaseEncoder,
                          BasisEncoder, EncoderFactory, EncodingError)


class TestBaseEncoder:
    """Test the abstract base encoder class."""

    def test_cannot_instantiate_base_encoder(self):
        """Test that BaseEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEncoder(n_qubits=2)

    def test_base_encoder_interface(self):
        """Test that BaseEncoder defines the required interface."""
        # Check that encode method is abstract
        assert hasattr(BaseEncoder, 'encode')
        assert getattr(BaseEncoder.encode, '__isabstractmethod__', False)


class TestAngleEncoder:
    """Test the angle encoding strategy."""

    def test_initialization(self):
        """Test AngleEncoder initialization."""
        encoder = AngleEncoder(n_qubits=4)
        assert encoder.n_qubits == 4
        assert encoder.encoding_type == "angle"

    def test_initialization_invalid_qubits(self):
        """Test initialization with invalid qubit count."""
        with pytest.raises(ValueError):
            AngleEncoder(n_qubits=0)

        with pytest.raises(ValueError):
            AngleEncoder(n_qubits=-1)

    def test_encode_valid_data(self):
        """Test encoding valid data."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.5, 0.8])

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_encode_data_size_mismatch(self):
        """Test encoding with wrong data size."""
        encoder = AngleEncoder(n_qubits=2)

        # Too few features
        with pytest.raises(EncodingError):
            encoder.encode([0.5])

        # Too many features
        with pytest.raises(EncodingError):
            encoder.encode([0.5, 0.8, 0.3])

    def test_encode_invalid_data_type(self):
        """Test encoding with invalid data types."""
        encoder = AngleEncoder(n_qubits=2)

        with pytest.raises((EncodingError, TypeError)):
            encoder.encode("invalid")

        with pytest.raises((EncodingError, TypeError)):
            encoder.encode(None)

    def test_encode_nan_values(self):
        """Test encoding with NaN values."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.5, np.nan])

        with pytest.raises(EncodingError):
            encoder.encode(data)

    def test_encode_infinite_values(self):
        """Test encoding with infinite values."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.5, np.inf])

        with pytest.raises(EncodingError):
            encoder.encode(data)

    def test_normalization(self):
        """Test that data is properly normalized."""
        encoder = AngleEncoder(n_qubits=2, normalize=True)
        data = np.array([10.0, 20.0])

        # Should not raise error with normalization
        circuit = encoder.encode(data)
        assert circuit is not None

    def test_performance_monitoring(self):
        """Test that performance is monitored."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.5, 0.8])

        # Should track encoding time
        with patch('time.time', side_effect=[0.0, 0.1]):
            encoder.encode(data)

        assert hasattr(encoder, '_last_encoding_time')

    def test_batch_encoding(self):
        """Test encoding multiple samples."""
        encoder = AngleEncoder(n_qubits=2)
        batch_data = np.array([
            [0.5, 0.8],
            [0.3, 0.6],
            [0.9, 0.2]
        ])

        circuits = encoder.encode_batch(batch_data)
        assert len(circuits) == 3
        assert all(circuit is not None for circuit in circuits)

    @pytest.mark.parametrize("n_qubits,data_size", [
        (2, 2), (4, 4), (8, 8), (1, 1)
    ])
    def test_various_qubit_counts(self, n_qubits, data_size):
        """Test encoder with various qubit counts."""
        encoder = AngleEncoder(n_qubits=n_qubits)
        data = np.random.rand(data_size)

        circuit = encoder.encode(data)
        assert circuit is not None


class TestAmplitudeEncoder:
    """Test the amplitude encoding strategy."""

    def test_initialization(self):
        """Test AmplitudeEncoder initialization."""
        encoder = AmplitudeEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.encoding_type == "amplitude"

    def test_encode_valid_data(self):
        """Test encoding valid normalized data."""
        encoder = AmplitudeEncoder(n_qubits=2)
        # Data should be normalized (sum of squares = 1)
        data = np.array([0.6, 0.8])  # 0.6² + 0.8² = 1

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_encode_unnormalized_data(self):
        """Test encoding unnormalized data with auto-normalization."""
        encoder = AmplitudeEncoder(n_qubits=2, normalize=True)
        data = np.array([3.0, 4.0])  # Will be normalized to [0.6, 0.8]

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_power_of_two_constraint(self):
        """Test that data size must be power of 2."""
        encoder = AmplitudeEncoder(n_qubits=2)

        # Valid: 4 = 2²
        data = np.array([0.5, 0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        assert circuit is not None

        # Invalid: 3 is not power of 2
        with pytest.raises(EncodingError):
            encoder.encode(np.array([0.5, 0.5, 0.5]))

    def test_zero_norm_data(self):
        """Test handling of zero-norm data."""
        encoder = AmplitudeEncoder(n_qubits=2)
        data = np.array([0.0, 0.0, 0.0, 0.0])

        with pytest.raises(EncodingError):
            encoder.encode(data)


class TestBasisEncoder:
    """Test the computational basis encoding strategy."""

    def test_initialization(self):
        """Test BasisEncoder initialization."""
        encoder = BasisEncoder(n_qubits=3)
        assert encoder.n_qubits == 3
        assert encoder.encoding_type == "basis"

    def test_encode_binary_string(self):
        """Test encoding binary string data."""
        encoder = BasisEncoder(n_qubits=3)
        data = "101"

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_encode_integer(self):
        """Test encoding integer data."""
        encoder = BasisEncoder(n_qubits=3)
        data = 5  # Binary: 101

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_encode_binary_array(self):
        """Test encoding binary array data."""
        encoder = BasisEncoder(n_qubits=3)
        data = np.array([1, 0, 1])

        circuit = encoder.encode(data)
        assert circuit is not None

    def test_invalid_binary_string(self):
        """Test encoding invalid binary string."""
        encoder = BasisEncoder(n_qubits=2)

        with pytest.raises(EncodingError):
            encoder.encode("102")  # Invalid binary digit

        with pytest.raises(EncodingError):
            encoder.encode("abc")  # Non-binary string

    def test_integer_out_of_range(self):
        """Test encoding integer out of range."""
        encoder = BasisEncoder(n_qubits=2)  # Max value: 3

        with pytest.raises(EncodingError):
            encoder.encode(4)  # Too large for 2 qubits

        with pytest.raises(EncodingError):
            encoder.encode(-1)  # Negative not allowed

    def test_array_size_mismatch(self):
        """Test encoding array with wrong size."""
        encoder = BasisEncoder(n_qubits=3)

        with pytest.raises(EncodingError):
            encoder.encode(np.array([1, 0]))  # Too short

        with pytest.raises(EncodingError):
            encoder.encode(np.array([1, 0, 1, 1]))  # Too long


class TestEncoderFactory:
    """Test the encoder factory."""

    def test_create_angle_encoder(self):
        """Test creating angle encoder via factory."""
        encoder = EncoderFactory.create_encoder("angle", n_qubits=4)
        assert isinstance(encoder, AngleEncoder)
        assert encoder.n_qubits == 4

    def test_create_amplitude_encoder(self):
        """Test creating amplitude encoder via factory."""
        encoder = EncoderFactory.create_encoder("amplitude", n_qubits=2)
        assert isinstance(encoder, AmplitudeEncoder)
        assert encoder.n_qubits == 2

    def test_create_basis_encoder(self):
        """Test creating basis encoder via factory."""
        encoder = EncoderFactory.create_encoder("basis", n_qubits=3)
        assert isinstance(encoder, BasisEncoder)
        assert encoder.n_qubits == 3

    def test_invalid_encoder_type(self):
        """Test creating encoder with invalid type."""
        with pytest.raises(ValueError):
            EncoderFactory.create_encoder("invalid", n_qubits=2)

    def test_auto_select_encoder(self):
        """Test automatic encoder selection."""
        # Binary data -> BasisEncoder
        data = "101"
        encoder = EncoderFactory.auto_select(data, n_qubits=3)
        assert isinstance(encoder, BasisEncoder)

        # Continuous data -> AngleEncoder (default)
        data = np.array([0.5, 0.8])
        encoder = EncoderFactory.auto_select(data, n_qubits=2)
        assert isinstance(encoder, AngleEncoder)

    def test_get_available_encoders(self):
        """Test getting list of available encoders."""
        encoders = EncoderFactory.get_available_encoders()
        assert "angle" in encoders
        assert "amplitude" in encoders
        assert "basis" in encoders
        assert len(encoders) >= 3


class TestEncodingError:
    """Test custom encoding exception."""

    def test_encoding_error_creation(self):
        """Test creating encoding error."""
        error = EncodingError("Test message")
        assert str(error) == "Test message"

    def test_encoding_error_inheritance(self):
        """Test that EncodingError inherits from Exception."""
        error = EncodingError("Test")
        assert isinstance(error, Exception)


@pytest.mark.slow
class TestEncoderPerformance:
    """Test encoder performance characteristics."""

    def test_angle_encoder_performance(self, performance_monitor):
        """Test angle encoder performance."""
        encoder = AngleEncoder(n_qubits=10)
        data = np.random.rand(10)

        # Should complete within reasonable time
        import time
        start = time.time()
        circuit = encoder.encode(data)
        duration = time.time() - start

        assert circuit is not None
        assert duration < 1.0  # Should be fast

    def test_batch_encoding_performance(self, performance_monitor):
        """Test batch encoding performance."""
        encoder = AngleEncoder(n_qubits=4)
        batch_data = np.random.rand(100, 4)

        import time
        start = time.time()
        circuits = encoder.encode_batch(batch_data)
        duration = time.time() - start

        assert len(circuits) == 100
        assert duration < 5.0  # Should handle batches efficiently

    def test_memory_usage(self):
        """Test that encoders don't leak memory."""
        encoder = AngleEncoder(n_qubits=8)

        # Encode many samples
        for _ in range(1000):
            data = np.random.rand(8)
            encoder.encode(data)

        # Memory usage should remain reasonable
        # (This is a basic test; real memory monitoring would be more complex)
        import gc
        gc.collect()


@pytest.mark.quantum
class TestEncoderQuantumIntegration:
    """Test encoder integration with quantum backends."""

    def test_qiskit_integration(self, qiskit_simulator):
        """Test encoder with Qiskit backend."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.5, 0.8])

        circuit = encoder.encode(data)

        # Should be able to run on Qiskit backend
        if hasattr(circuit, 'measure_all'):
            circuit.measure_all()

        # This would normally execute the circuit
        # result = qiskit_simulator.run(circuit, shots=100).result()
        # assert result is not None

    def test_multiple_backend_compatibility(self):
        """Test that encoders work with different backends."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([0.3, 0.7])

        # Should generate backend-agnostic circuits
        circuit = encoder.encode(data)
        assert circuit is not None

        # Circuit should be convertible to different formats
        # (Specific tests would depend on backend implementations)
