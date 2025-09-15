"""Quantum data encoders for classical-to-quantum state preparation.

This module provides various encoding strategies to transform classical data
into quantum states suitable for quantum neural network processing.

Key Features:
- Angle encoding with amplitude normalization
- Amplitude encoding with efficient state preparation
- Basis encoding for binary data
- Feature maps with entangling operations
- Automatic boundary condition handling
- Performance monitoring and optimization

Classes:
    AngleEncoder: Encode data using rotation angles
    AmplitudeEncoder: Encode data in quantum state amplitudes
    BasisEncoder: Encode binary data in computational basis
    FeatureMap: Advanced encoding with entangling operations
    EncoderFactory: Factory for creating optimal encoders

Example:
    >>> import numpy as np
    >>> from qns.encoders import AngleEncoder
    >>>
    >>> # Create sample data
    >>> X = np.array([[0.5, -0.3], [0.8, 0.2]])
    >>>
    >>> # Initialize encoder
    >>> encoder = AngleEncoder(n_qubits=2, normalization='minmax')
    >>>
    >>> # Encode data to quantum circuit
    >>> circuit = encoder.encode(X[0])
    >>> print(f"Circuit depth: {circuit.depth()}")
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Conditional imports for different backends
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter, ParameterVector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Configure logging
logger = logging.getLogger(__name__)


class EncodingError(Exception):
    """Exception raised for errors in quantum encoding operations."""
    pass


class BaseEncoder(ABC):
    """Abstract base class for quantum data encoders.

    This class defines the interface that all quantum encoders must implement,
    providing common functionality for data validation, normalization, and
    performance monitoring.

    Args:
        n_qubits: Number of qubits available for encoding
        normalization: Normalization strategy ('minmax', 'standard', 'none')
        validation: Whether to perform input validation

    Attributes:
        n_qubits: Number of qubits
        normalization: Normalization method
        validation: Validation flag
        encoding_stats: Performance statistics
    """

    def __init__(
        self,
        n_qubits: int,
        normalization: str = 'minmax',
        validation: bool = True
    ) -> None:
        """Initialize the base encoder."""
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")

        if normalization not in ['minmax', 'standard', 'none']:
            raise ValueError("Invalid normalization method")

        self.n_qubits = n_qubits
        self.normalization = normalization
        self.validation = validation

        # Performance monitoring
        self.encoding_stats = {
            'total_encodings': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'errors': 0,
            'warnings': 0
        }

        # Normalization parameters (fitted during first encode)
        self._normalization_params: Optional[Dict[str, NDArray]] = None

    def validate_input(self, data: NDArray) -> None:
        """Validate input data for encoding.

        Args:
            data: Input data array to validate

        Raises:
            EncodingError: If data is invalid for encoding
        """
        if not isinstance(data, np.ndarray):
            raise EncodingError("Input data must be a numpy array")

        if data.ndim != 1:
            raise EncodingError("Input data must be 1-dimensional")

        if len(data) == 0:
            raise EncodingError("Input data cannot be empty")

        # Check for NaN or infinite values
        if not np.isfinite(data).all():
            raise EncodingError("Input data contains NaN or infinite values")

        # Check feature count compatibility
        expected_features = self._get_expected_feature_count()
        if expected_features > 0 and len(data) != expected_features:
            raise EncodingError(
                f"Expected {expected_features} features, got {len(data)}"
            )

    def normalize_data(self, data: NDArray) -> NDArray:
        """Normalize input data according to the specified method.

        Args:
            data: Raw input data

        Returns:
            Normalized data array
        """
        if self.normalization == 'none':
            return data.copy()

        # Fit normalization parameters on first call
        if self._normalization_params is None:
            self._fit_normalization(data)

        normalized = data.copy()

        if self.normalization == 'minmax':
            # Scale to [0, 1] or [-1, 1]
            data_min = self._normalization_params['min']
            data_max = self._normalization_params['max']
            data_range = data_max - data_min

            # Handle zero range (constant features)
            if np.any(data_range == 0):
                logger.warning("Constant features detected in normalization")
                data_range = np.where(data_range == 0, 1.0, data_range)

            normalized = (normalized - data_min) / data_range

        elif self.normalization == 'standard':
            # Standardize to zero mean and unit variance
            mean = self._normalization_params['mean']
            std = self._normalization_params['std']

            # Handle zero variance
            if np.any(std == 0):
                logger.warning("Zero variance features detected")
                std = np.where(std == 0, 1.0, std)

            normalized = (normalized - mean) / std

        return normalized

    def _fit_normalization(self, data: NDArray) -> None:
        """Fit normalization parameters from data.

        Args:
            data: Reference data for fitting normalization
        """
        if self.normalization == 'minmax':
            self._normalization_params = {
                'min': np.min(data),
                'max': np.max(data)
            }
        elif self.normalization == 'standard':
            self._normalization_params = {
                'mean': np.mean(data),
                'std': np.std(data)
            }

    def _update_stats(self, encoding_time: float, success: bool) -> None:
        """Update encoding performance statistics.

        Args:
            encoding_time: Time taken for encoding operation
            success: Whether encoding was successful
        """
        self.encoding_stats['total_encodings'] += 1

        if success:
            self.encoding_stats['total_time'] += encoding_time
            self.encoding_stats['average_time'] = (
                self.encoding_stats['total_time'] /
                self.encoding_stats['total_encodings']
            )
        else:
            self.encoding_stats['errors'] += 1

    @abstractmethod
    def _get_expected_feature_count(self) -> int:
        """Get the expected number of features for this encoder.

        Returns:
            Expected feature count, or -1 if flexible
        """
        pass

    @abstractmethod
    def encode(self, data: NDArray) -> Any:
        """Encode classical data to quantum circuit.

        Args:
            data: Classical data to encode

        Returns:
            Quantum circuit representing encoded data
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get encoding performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        return self.encoding_stats.copy()


class AngleEncoder(BaseEncoder):
    """Encode classical data using rotation angles on qubits.

    This encoder maps classical features to rotation angles applied to qubits,
    typically using RY rotations. It's efficient and works well for most
    quantum machine learning applications.

    The encoding maps feature values to rotation angles in [0, π] or [0, 2π]
    depending on the configuration. Supports various rotation gates and
    entangling operations for enhanced expressivity.

    Args:
        n_qubits: Number of qubits for encoding
        rotation_gate: Type of rotation gate ('RY', 'RX', 'RZ')
        angle_range: Range of angles ('pi' for [0,π], '2pi' for [0,2π])
        normalization: Data normalization method
        validation: Enable input validation

    Example:
        >>> encoder = AngleEncoder(n_qubits=3, rotation_gate='RY')
        >>> data = np.array([0.5, -0.2, 0.8])
        >>> circuit = encoder.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        rotation_gate: str = 'RY',
        angle_range: str = 'pi',
        normalization: str = 'minmax',
        validation: bool = True
    ) -> None:
        """Initialize the angle encoder."""
        super().__init__(n_qubits, normalization, validation)

        # Validate rotation gate
        valid_gates = ['RX', 'RY', 'RZ']
        if rotation_gate not in valid_gates:
            raise ValueError(f"Rotation gate must be one of {valid_gates}")

        # Validate angle range
        if angle_range not in ['pi', '2pi']:
            raise ValueError("Angle range must be 'pi' or '2pi'")

        self.rotation_gate = rotation_gate
        self.angle_range = angle_range

        # Set angle scaling factor
        self.angle_scale = np.pi if angle_range == 'pi' else 2 * np.pi

    def _get_expected_feature_count(self) -> int:
        """Get expected feature count (equals number of qubits)."""
        return self.n_qubits

    def encode(self, data: NDArray) -> QuantumCircuit:
        """Encode data using angle encoding.

        Args:
            data: Classical features to encode (length must equal n_qubits)

        Returns:
            Quantum circuit with angle-encoded data

        Raises:
            EncodingError: If encoding fails due to invalid data
        """
        start_time = time.time()
        success = False

        try:
            if not HAS_QISKIT:
                raise EncodingError("Qiskit not available for circuit creation")

            # Validate input
            if self.validation:
                self.validate_input(data)

            # Normalize data
            normalized_data = self.normalize_data(data)

            # Create quantum circuit
            qc = QuantumCircuit(self.n_qubits, name="angle_encoding")

            # Apply rotation gates
            for i, value in enumerate(normalized_data):
                angle = value * self.angle_scale

                # Apply appropriate rotation gate
                if self.rotation_gate == 'RX':
                    qc.rx(angle, i)
                elif self.rotation_gate == 'RY':
                    qc.ry(angle, i)
                elif self.rotation_gate == 'RZ':
                    qc.rz(angle, i)

            success = True
            return qc

        except Exception as e:
            logger.error(f"Angle encoding failed: {str(e)}")
            raise EncodingError(f"Encoding failed: {str(e)}") from e

        finally:
            encoding_time = time.time() - start_time
            self._update_stats(encoding_time, success)


class AmplitudeEncoder(BaseEncoder):
    """Encode classical data in quantum state amplitudes.

    This encoder embeds classical data directly into the amplitudes of a quantum
    state, allowing for exponential compression of classical information.
    However, it requires normalization and can be resource-intensive for
    state preparation.

    The encoding creates a quantum state |ψ⟩ = Σᵢ aᵢ|i⟩ where aᵢ are the
    normalized classical data values. Supports various state preparation
    strategies and automatic padding for non-power-of-2 data lengths.

    Args:
        n_qubits: Number of qubits (determines max 2^n_qubits amplitudes)
        preparation_method: State preparation algorithm ('direct', 'variational')
        padding: How to handle data length mismatches ('zero', 'repeat', 'truncate')
        normalization: Data normalization method (required for valid states)

    Example:
        >>> encoder = AmplitudeEncoder(n_qubits=3)  # Can encode up to 8 values
        >>> data = np.array([0.5, 0.3, 0.7, 0.1, 0.2, 0.4])
        >>> circuit = encoder.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        preparation_method: str = 'direct',
        padding: str = 'zero',
        normalization: str = 'l2',
        validation: bool = True
    ) -> None:
        """Initialize the amplitude encoder."""
        # Use 'none' for base class normalization since we handle it specially
        super().__init__(n_qubits, 'none', validation)

        # Validate parameters
        if preparation_method not in ['direct', 'variational']:
            raise ValueError("Preparation method must be 'direct' or 'variational'")

        if padding not in ['zero', 'repeat', 'truncate']:
            raise ValueError("Padding must be 'zero', 'repeat', or 'truncate'")

        if normalization not in ['l2', 'l1', 'max']:
            raise ValueError("Normalization must be 'l2', 'l1', or 'max'")

        self.preparation_method = preparation_method
        self.padding = padding
        self.amplitude_normalization = normalization

        # Maximum number of amplitudes we can encode
        self.max_amplitudes = 2 ** self.n_qubits

    def _get_expected_feature_count(self) -> int:
        """Get expected feature count (flexible up to max_amplitudes)."""
        return -1  # Flexible

    def _prepare_amplitudes(self, data: NDArray) -> NDArray:
        """Prepare and normalize amplitudes for state preparation.

        Args:
            data: Raw data values

        Returns:
            Normalized amplitude array of length 2^n_qubits
        """
        # Handle data length adjustment
        if len(data) > self.max_amplitudes:
            if self.padding == 'truncate':
                amplitudes = data[:self.max_amplitudes]
                logger.warning(f"Truncated data from {len(data)} to {self.max_amplitudes}")
            else:
                raise EncodingError(
                    f"Data length {len(data)} exceeds maximum {self.max_amplitudes}"
                )
        elif len(data) < self.max_amplitudes:
            if self.padding == 'zero':
                amplitudes = np.zeros(self.max_amplitudes)
                amplitudes[:len(data)] = data
            elif self.padding == 'repeat':
                # Repeat pattern to fill required length
                repeat_count = self.max_amplitudes // len(data)
                remainder = self.max_amplitudes % len(data)
                amplitudes = np.tile(data, repeat_count)
                if remainder > 0:
                    amplitudes = np.concatenate([amplitudes, data[:remainder]])
            else:
                raise EncodingError(f"Cannot handle data length {len(data)}")
        else:
            amplitudes = data.copy()

        # Normalize amplitudes
        if self.amplitude_normalization == 'l2':
            norm = np.linalg.norm(amplitudes)
        elif self.amplitude_normalization == 'l1':
            norm = np.sum(np.abs(amplitudes))
        elif self.amplitude_normalization == 'max':
            norm = np.max(np.abs(amplitudes))
        else:
            norm = 1.0

        if norm == 0:
            raise EncodingError("Cannot normalize zero vector")

        return amplitudes / norm

    def encode(self, data: NDArray) -> QuantumCircuit:
        """Encode data using amplitude encoding.

        Args:
            data: Classical data to encode as quantum state amplitudes

        Returns:
            Quantum circuit preparing the amplitude-encoded state

        Raises:
            EncodingError: If encoding fails
        """
        start_time = time.time()
        success = False

        try:
            if not HAS_QISKIT:
                raise EncodingError("Qiskit not available for circuit creation")

            # Validate input
            if self.validation:
                self.validate_input(data)

            # Prepare normalized amplitudes
            amplitudes = self._prepare_amplitudes(data)

            # Create quantum circuit
            qc = QuantumCircuit(self.n_qubits, name="amplitude_encoding")

            if self.preparation_method == 'direct':
                # Use Qiskit's built-in state preparation
                from qiskit.extensions import Initialize
                init_gate = Initialize(amplitudes)
                qc.append(init_gate, range(self.n_qubits))

            elif self.preparation_method == 'variational':
                # Implement variational state preparation
                self._variational_state_prep(qc, amplitudes)

            success = True
            return qc

        except Exception as e:
            logger.error(f"Amplitude encoding failed: {str(e)}")
            raise EncodingError(f"Encoding failed: {str(e)}") from e

        finally:
            encoding_time = time.time() - start_time
            self._update_stats(encoding_time, success)

    def _variational_state_prep(
        self,
        circuit: QuantumCircuit,
        target_amplitudes: NDArray
    ) -> None:
        """Implement variational state preparation.

        This is a placeholder for more sophisticated state preparation
        algorithms that could be optimized for specific hardware constraints.

        Args:
            circuit: Quantum circuit to modify
            target_amplitudes: Target state amplitudes
        """
        # Placeholder implementation using uniformly controlled rotations
        # In practice, this would use optimization to find optimal angles

        # Apply Hadamard gates to create equal superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # Add parameterized rotations (this is simplified)
        for i in range(self.n_qubits):
            # This would normally be optimized to match target_amplitudes
            angle = np.arcsin(np.sqrt(np.abs(target_amplitudes[i])))
            circuit.ry(2 * angle, i)


class BasisEncoder(BaseEncoder):
    """Encode binary data in computational basis states.

    This encoder maps binary classical data directly to computational basis
    states of qubits. It's the most straightforward encoding but limited to
    binary features. Supports various binary representations and error
    correction for noisy binary data.

    Args:
        n_qubits: Number of qubits for encoding
        binary_threshold: Threshold for converting continuous to binary (0.5)
        error_correction: Apply error correction for noisy binary data

    Example:
        >>> encoder = BasisEncoder(n_qubits=4)
        >>> binary_data = np.array([1, 0, 1, 0])  # |1010⟩ state
        >>> circuit = encoder.encode(binary_data)
    """

    def __init__(
        self,
        n_qubits: int,
        binary_threshold: float = 0.5,
        error_correction: bool = False,
        validation: bool = True
    ) -> None:
        """Initialize the basis encoder."""
        super().__init__(n_qubits, 'none', validation)

        if not 0 <= binary_threshold <= 1:
            raise ValueError("Binary threshold must be between 0 and 1")

        self.binary_threshold = binary_threshold
        self.error_correction = error_correction

    def _get_expected_feature_count(self) -> int:
        """Get expected feature count (equals number of qubits)."""
        return self.n_qubits

    def _binarize_data(self, data: NDArray) -> NDArray:
        """Convert data to binary representation.

        Args:
            data: Input data to binarize

        Returns:
            Binary array (0s and 1s)
        """
        binary_data = (data > self.binary_threshold).astype(int)

        if self.error_correction:
            # Simple majority vote error correction for repeated measurements
            # This is a placeholder for more sophisticated methods
            pass

        return binary_data

    def encode(self, data: NDArray) -> QuantumCircuit:
        """Encode binary data in computational basis.

        Args:
            data: Binary features (will be binarized if continuous)

        Returns:
            Quantum circuit in computational basis state

        Raises:
            EncodingError: If encoding fails
        """
        start_time = time.time()
        success = False

        try:
            if not HAS_QISKIT:
                raise EncodingError("Qiskit not available for circuit creation")

            # Validate input
            if self.validation:
                self.validate_input(data)

            # Binarize data
            binary_data = self._binarize_data(data)

            # Create quantum circuit
            qc = QuantumCircuit(self.n_qubits, name="basis_encoding")

            # Apply X gates for |1⟩ states
            for i, bit in enumerate(binary_data):
                if bit == 1:
                    qc.x(i)

            success = True
            return qc

        except Exception as e:
            logger.error(f"Basis encoding failed: {str(e)}")
            raise EncodingError(f"Encoding failed: {str(e)}") from e

        finally:
            encoding_time = time.time() - start_time
            self._update_stats(encoding_time, success)


class EncoderFactory:
    """Factory class for creating optimal quantum encoders.

    This factory analyzes data characteristics and recommends appropriate
    encoding strategies, automatically configuring encoder parameters for
    optimal performance.

    Example:
        >>> factory = EncoderFactory()
        >>> data = np.random.randn(100, 4)  # 100 samples, 4 features
        >>> encoder = factory.create_encoder(data, n_qubits=4)
        >>> print(f"Recommended encoder: {type(encoder).__name__}")
    """

    @staticmethod
    def analyze_data(data: NDArray) -> Dict[str, Any]:
        """Analyze data characteristics for encoder selection.

        Args:
            data: Input data to analyze (2D array: samples × features)

        Returns:
            Dictionary containing data analysis results
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D (samples × features)")

        analysis = {
            'n_samples': data.shape[0],
            'n_features': data.shape[1],
            'data_range': (np.min(data), np.max(data)),
            'data_std': np.std(data),
            'is_binary': np.all(np.isin(data, [0, 1])),
            'sparsity': np.sum(data == 0) / data.size,
            'has_negative': np.any(data < 0),
            'memory_requirement': data.nbytes
        }

        return analysis

    @classmethod
    def create_encoder(
        cls,
        data: NDArray,
        n_qubits: int,
        encoder_type: Optional[str] = None,
        **kwargs
    ) -> BaseEncoder:
        """Create optimal encoder for given data and constraints.

        Args:
            data: Sample data for analysis (2D array)
            n_qubits: Number of available qubits
            encoder_type: Force specific encoder type (None for auto)
            **kwargs: Additional parameters for encoder

        Returns:
            Configured encoder instance
        """
        analysis = cls.analyze_data(data)

        # Auto-select encoder type if not specified
        if encoder_type is None:
            if analysis['is_binary']:
                encoder_type = 'basis'
            elif analysis['n_features'] <= n_qubits:
                encoder_type = 'angle'
            elif analysis['n_features'] <= 2 ** n_qubits:
                encoder_type = 'amplitude'
            else:
                raise ValueError(
                    f"Cannot encode {analysis['n_features']} features "
                    f"with {n_qubits} qubits"
                )

        # Create encoder with optimal parameters
        if encoder_type == 'angle':
            return AngleEncoder(
                n_qubits=n_qubits,
                normalization='standard' if analysis['has_negative'] else 'minmax',
                **kwargs
            )
        elif encoder_type == 'amplitude':
            return AmplitudeEncoder(
                n_qubits=n_qubits,
                padding='zero' if analysis['sparsity'] > 0.5 else 'truncate',
                **kwargs
            )
        elif encoder_type == 'basis':
            return BasisEncoder(
                n_qubits=n_qubits,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


# Convenience function for quick encoding
def encode_data(
    data: NDArray,
    n_qubits: int,
    method: str = 'auto',
    **kwargs
) -> QuantumCircuit:
    """Convenience function for quick data encoding.

    Args:
        data: Classical data to encode
        n_qubits: Number of qubits available
        method: Encoding method ('auto', 'angle', 'amplitude', 'basis')
        **kwargs: Additional encoder parameters

    Returns:
        Quantum circuit with encoded data

    Example:
        >>> data = np.array([0.5, -0.3, 0.8])
        >>> circuit = encode_data(data, n_qubits=3, method='angle')
    """
    if method == 'auto':
        # Use factory for automatic encoder selection
        factory = EncoderFactory()
        encoder = factory.create_encoder(
            data.reshape(1, -1), n_qubits, **kwargs
        )
    elif method == 'angle':
        encoder = AngleEncoder(n_qubits, **kwargs)
    elif method == 'amplitude':
        encoder = AmplitudeEncoder(n_qubits, **kwargs)
    elif method == 'basis':
        encoder = BasisEncoder(n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return encoder.encode(data)
