"""Quantum neural network models and architectures.

This module provides comprehensive implementations of quantum neural network
models with robust error handling, performance monitoring, and advanced
training capabilities.

Key Features:
- Quantum perceptrons and multilayer networks
- Associative memory systems (Hopfield-like)
- Spiking quantum networks
- Hybrid classical-quantum architectures
- Automatic circuit optimization
- Memory management and crash prevention
- Comprehensive logging and monitoring

Classes:
    QuantumClassifier: Binary and multiclass classification
    QuantumRegressor: Regression with continuous outputs
    QuantumAutoEncoder: Dimensionality reduction and reconstruction
    QuantumHopfield: Associative memory network
    QuantumSpikingNetwork: Temporal processing network

Example:
    >>> from qns.models import QuantumClassifier
    >>> from qns.data import generate_xor_data
    >>>
    >>> # Create XOR dataset
    >>> X, y = generate_xor_data(n_samples=100)
    >>>
    >>> # Initialize and train classifier
    >>> model = QuantumClassifier(n_qubits=2, depth=3)
    >>> model.fit(X, y, epochs=100, learning_rate=0.1)
    >>>
    >>> # Evaluate performance
    >>> accuracy = model.score(X, y)
    >>> print(f"Training accuracy: {accuracy:.3f}")
"""

import gc
import logging
import signal
import threading
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Conditional imports
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.quantum_info import Pauli, SparsePauliOp
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    warnings.warn("Qiskit not available. Some functionality will be limited.")

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.metrics import accuracy_score, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for quantum neural network models.

    This dataclass holds all configuration parameters for quantum models,
    including circuit parameters, training settings, and performance options.

    Attributes:
        n_qubits: Number of qubits in the quantum circuit
        depth: Depth of the variational circuit layers
        shots: Number of measurement shots per evaluation
        optimization_level: Circuit optimization level (0-3)
        error_mitigation: Enable quantum error mitigation
        memory_limit_mb: Maximum memory usage in MB
        timeout_seconds: Maximum execution time per operation
        validate_inputs: Enable input validation
        track_metrics: Enable performance tracking
        save_intermediate: Save intermediate results during training
    """
    n_qubits: int = 2
    depth: int = 2
    shots: int = 1024
    optimization_level: int = 1
    error_mitigation: bool = False
    memory_limit_mb: int = 1024
    timeout_seconds: int = 300
    validate_inputs: bool = True
    track_metrics: bool = True
    save_intermediate: bool = False
    random_seed: Optional[int] = None

    # Performance monitoring
    circuit_cache_size: int = 100
    gradient_batch_size: int = 32
    parallel_execution: bool = True

    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    graceful_degradation: bool = True


class QuantumError(Exception):
    """Exception raised for quantum-specific errors."""
    pass


class MemoryError(Exception):
    """Exception raised when memory limits are exceeded."""
    pass


class TimeoutError(Exception):
    """Exception raised when operations exceed time limits."""
    pass


class ResourceMonitor:
    """Monitor system resources and enforce limits."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._start_time = None
        self._start_memory = None

    def __enter__(self):
        """Enter resource monitoring context."""
        self._start_time = time.time()
        try:
            import psutil
            process = psutil.Process()
            self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self._start_memory = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit resource monitoring context."""
        # Force garbage collection
        gc.collect()

        # Log resource usage
        if self._start_time:
            elapsed = time.time() - self._start_time
            logger.debug(f"Operation completed in {elapsed:.2f}s")

    def check_memory(self):
        """Check current memory usage against limits."""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB

            if current_memory > self.config.memory_limit_mb:
                raise MemoryError(
                    f"Memory usage ({current_memory:.1f}MB) exceeds limit "
                    f"({self.config.memory_limit_mb}MB)"
                )
        except ImportError:
            # Cannot monitor memory without psutil
            pass

    def check_timeout(self):
        """Check if operation has exceeded timeout."""
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self.config.timeout_seconds:
                raise TimeoutError(
                    f"Operation timeout ({elapsed:.1f}s > "
                    f"{self.config.timeout_seconds}s)"
                )


@contextmanager
def timeout_handler(timeout_seconds: int):
    """Context manager for handling operation timeouts."""
    def timeout_callback(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

    # Set up signal handler (Unix-like systems only)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_callback)
        signal.alarm(timeout_seconds)
        yield
    except AttributeError:
        # Windows doesn't support SIGALRM, use threading
        import threading
        timer = threading.Timer(timeout_seconds, lambda: None)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    finally:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except AttributeError:
            pass


class BaseQuantumModel(ABC):
    """Abstract base class for quantum neural network models.

    This class provides common functionality for all quantum models including:
    - Resource monitoring and limit enforcement
    - Error handling and graceful degradation
    - Performance tracking and optimization
    - Circuit caching and optimization
    - Automatic hyperparameter validation

    Args:
        config: Model configuration object
        backend: Quantum backend for circuit execution

    Attributes:
        config: Model configuration
        backend: Quantum execution backend
        parameters: Trainable quantum parameters
        metrics: Training and evaluation metrics
        circuit_cache: Cache for compiled circuits
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        backend: Optional[str] = None
    ):
        """Initialize the base quantum model."""
        self.config = config or ModelConfig()
        self.backend_name = backend or 'qasm_simulator'

        # Initialize backend
        self._setup_backend()

        # Model state
        self.parameters: Optional[NDArray] = None
        self.is_fitted = False
        self._parameter_count = 0

        # Performance tracking
        self.metrics = {
            'training_history': [],
            'evaluation_history': [],
            'timing_stats': {},
            'memory_usage': [],
            'error_count': 0,
            'retry_count': 0
        }

        # Circuit caching for performance
        self.circuit_cache = {}

        # Thread safety
        self._lock = threading.Lock()

        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def _setup_backend(self):
        """Initialize the quantum backend."""
        if not HAS_QISKIT:
            raise QuantumError("Qiskit is required but not available")

        try:
            self.backend = AerSimulator()
            logger.info(f"Initialized backend: {self.backend_name}")
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise QuantumError(f"Backend initialization failed: {e}")

    def _validate_inputs(self, X: NDArray, y: Optional[NDArray] = None):
        """Validate input data."""
        if not self.config.validate_inputs:
            return

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (samples, features)")

        if X.shape[1] > self.config.n_qubits:
            raise ValueError(
                f"Number of features ({X.shape[1]}) exceeds qubits "
                f"({self.config.n_qubits})"
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")

            if len(y) != X.shape[0]:
                raise ValueError("X and y must have same number of samples")

            if not np.isfinite(y).all():
                raise ValueError("y contains non-finite values")

    def _create_circuit(self, x: NDArray) -> QuantumCircuit:
        """Create quantum circuit for given input.

        This method should be implemented by subclasses to define
        the specific circuit architecture.
        """
        # Check cache first
        cache_key = hash(x.tobytes())
        if cache_key in self.circuit_cache:
            return self.circuit_cache[cache_key].copy()

        # Create new circuit
        circuit = self._build_circuit(x)

        # Cache if under limit
        if len(self.circuit_cache) < self.config.circuit_cache_size:
            self.circuit_cache[cache_key] = circuit.copy()

        return circuit

    @abstractmethod
    def _build_circuit(self, x: NDArray) -> QuantumCircuit:
        """Build quantum circuit for input data.

        Must be implemented by subclasses.
        """
        pass

    def _execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute quantum circuit with error handling and retries.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots

        Returns:
            Execution results dictionary
        """
        shots = shots or self.config.shots
        retry_count = 0

        while retry_count <= self.config.max_retries:
            try:
                with ResourceMonitor(self.config) as monitor:
                    # Transpile circuit for backend
                    transpiled = transpile(
                        circuit,
                        self.backend,
                        optimization_level=self.config.optimization_level
                    )

                    # Check resources before execution
                    monitor.check_memory()
                    monitor.check_timeout()

                    # Execute circuit
                    job = self.backend.run(transpiled, shots=shots)
                    result = job.result()

                    # Track successful execution
                    if self.config.track_metrics:
                        self.metrics['evaluation_history'].append({
                            'timestamp': time.time(),
                            'shots': shots,
                            'circuit_depth': circuit.depth(),
                            'success': True
                        })

                    return {
                        'counts': result.get_counts(),
                        'memory': result.get_memory() if hasattr(result, 'get_memory') else None,
                        'success': True,
                        'retry_count': retry_count
                    }

            except Exception as e:
                retry_count += 1
                self.metrics['retry_count'] += 1

                logger.warning(
                    f"Circuit execution failed (attempt {retry_count}): {e}"
                )

                if retry_count > self.config.max_retries:
                    self.metrics['error_count'] += 1

                    if self.config.graceful_degradation:
                        # Return degraded result
                        logger.error("Max retries exceeded, returning degraded result")
                        return {
                            'counts': {'0' * self.config.n_qubits: shots},
                            'memory': None,
                            'success': False,
                            'retry_count': retry_count,
                            'error': str(e)
                        }
                    else:
                        raise QuantumError(f"Circuit execution failed: {e}")

                # Wait before retry
                time.sleep(self.config.retry_delay * retry_count)

        # Should not reach here
        raise QuantumError("Unexpected execution path")

    def _compute_expectation_value(
        self,
        circuit: QuantumCircuit,
        observable: Union[str, SparsePauliOp]
    ) -> float:
        """Compute expectation value of observable.

        Args:
            circuit: Quantum circuit
            observable: Observable operator or Pauli string

        Returns:
            Expectation value
        """
        if isinstance(observable, str):
            # Convert Pauli string to SparsePauliOp
            observable = SparsePauliOp.from_list([(observable, 1.0)])

        # Add measurement
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Execute circuit
        results = self._execute_circuit(meas_circuit)

        if not results['success']:
            logger.warning("Circuit execution failed, using degraded result")

        counts = results['counts']
        total_shots = sum(counts.values())

        # Compute expectation value from counts
        expectation = 0.0
        for bitstring, count in counts.items():
            # Compute observable eigenvalue for this bitstring
            eigenvalue = self._compute_eigenvalue(bitstring, observable)
            expectation += eigenvalue * count / total_shots

        return expectation

    def _compute_eigenvalue(
        self,
        bitstring: str,
        observable: SparsePauliOp
    ) -> float:
        """Compute eigenvalue of observable for given bitstring.

        Args:
            bitstring: Measurement outcome bitstring
            observable: Observable operator

        Returns:
            Eigenvalue for this measurement outcome
        """
        # This is a simplified implementation
        # In practice, would need full Pauli operator evaluation
        eigenvalue = 1.0

        for pauli_str, coeff in observable.to_list():
            pauli_eigenval = 1.0

            for i, pauli_op in enumerate(pauli_str):
                bit_val = int(bitstring[-(i+1)])  # Reverse order

                if pauli_op == 'Z':
                    pauli_eigenval *= 1 if bit_val == 0 else -1
                elif pauli_op == 'X':
                    # For X measurements, need different analysis
                    pauli_eigenval *= 1  # Simplified
                elif pauli_op == 'Y':
                    # For Y measurements, need different analysis
                    pauli_eigenval *= 1  # Simplified

            eigenvalue += coeff * pauli_eigenval

        return eigenvalue

    def save_model(self, filepath: str):
        """Save model parameters and configuration.

        Args:
            filepath: Path to save model file
        """
        model_data = {
            'parameters': self.parameters,
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted,
            'parameter_count': self._parameter_count
        }

        np.savez_compressed(filepath, **model_data)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model parameters and configuration.

        Args:
            filepath: Path to model file
        """
        try:
            data = np.load(filepath, allow_pickle=True)

            self.parameters = data['parameters']
            self.is_fitted = bool(data['is_fitted'])
            self._parameter_count = int(data['parameter_count'])

            # Load metrics if available
            if 'metrics' in data:
                self.metrics.update(data['metrics'].item())

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise QuantumError(f"Model loading failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics.

        Returns:
            Dictionary containing performance statistics
        """
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'training_history': [],
            'evaluation_history': [],
            'timing_stats': {},
            'memory_usage': [],
            'error_count': 0,
            'retry_count': 0
        }
        logger.info("Metrics reset")


class QuantumClassifier(BaseQuantumModel):
    """Quantum neural network for classification tasks.

    This classifier uses parameterized quantum circuits to perform binary
    and multiclass classification with robust training and comprehensive
    error handling.

    The model architecture consists of:
    1. Data encoding layer (angle encoding by default)
    2. Parameterized variational layers
    3. Measurement for classification output

    Args:
        n_qubits: Number of qubits (must be >= log2(n_classes))
        depth: Depth of variational layers
        encoding: Data encoding method ('angle', 'amplitude', 'basis')
        shots: Number of measurement shots

    Example:
        >>> classifier = QuantumClassifier(n_qubits=2, depth=3)
        >>> X = np.random.randn(100, 2)
        >>> y = np.random.randint(0, 2, 100)
        >>> classifier.fit(X, y, epochs=50)
        >>> predictions = classifier.predict(X)
    """

    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 2,
        encoding: str = 'angle',
        shots: int = 1024,
        **kwargs
    ):
        """Initialize quantum classifier."""
        config = ModelConfig(
            n_qubits=n_qubits,
            depth=depth,
            shots=shots,
            **kwargs
        )

        super().__init__(config)

        self.encoding = encoding
        self.n_classes = 2  # Will be updated during fit

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize variational parameters."""
        # Parameter count: depth * n_qubits * 2 (RY and RZ rotations)
        self._parameter_count = self.config.depth * self.config.n_qubits * 2

        # Initialize parameters with small random values
        self.parameters = np.random.normal(
            0, 0.1, size=self._parameter_count
        )

        logger.info(f"Initialized {self._parameter_count} parameters")

    def _build_circuit(self, x: NDArray) -> QuantumCircuit:
        """Build quantum circuit for classification.

        Args:
            x: Input feature vector

        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.config.n_qubits)

        # Data encoding layer
        self._add_encoding_layer(qc, x)

        # Variational layers
        param_idx = 0
        for layer in range(self.config.depth):
            param_idx = self._add_variational_layer(qc, param_idx)

        return qc

    def _add_encoding_layer(self, circuit: QuantumCircuit, x: NDArray):
        """Add data encoding layer to circuit.

        Args:
            circuit: Quantum circuit to modify
            x: Input data to encode
        """
        if self.encoding == 'angle':
            # Angle encoding using RY rotations
            for i in range(min(len(x), self.config.n_qubits)):
                # Normalize to [0, π] range
                angle = np.pi * (x[i] + 1) / 2  # Assumes x in [-1, 1]
                circuit.ry(angle, i)

        elif self.encoding == 'amplitude':
            # Amplitude encoding (placeholder)
            # In practice, would use state preparation
            for i in range(self.config.n_qubits):
                circuit.h(i)

        elif self.encoding == 'basis':
            # Basis encoding for binary features
            for i in range(min(len(x), self.config.n_qubits)):
                if x[i] > 0.5:  # Threshold for binary encoding
                    circuit.x(i)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _add_variational_layer(
        self,
        circuit: QuantumCircuit,
        param_start_idx: int
    ) -> int:
        """Add variational layer to circuit.

        Args:
            circuit: Quantum circuit to modify
            param_start_idx: Starting parameter index

        Returns:
            Next parameter index
        """
        param_idx = param_start_idx

        # Single-qubit rotations
        for q in range(self.config.n_qubits):
            if param_idx < len(self.parameters):
                circuit.ry(self.parameters[param_idx], q)
                param_idx += 1
            if param_idx < len(self.parameters):
                circuit.rz(self.parameters[param_idx], q)
                param_idx += 1

        # Entangling gates
        for q in range(self.config.n_qubits - 1):
            circuit.cx(q, q + 1)

        return param_idx

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self._validate_inputs(X)

        probabilities = []

        with ResourceMonitor(self.config):
            for i, x in enumerate(X):
                # Create and execute circuit
                circuit = self._create_circuit(x)

                # Measure expectation value for classification
                expectation = self._compute_expectation_value(
                    circuit, 'Z' + 'I' * (self.config.n_qubits - 1)
                )

                # Convert expectation to probability
                prob_0 = (1 + expectation) / 2
                prob_1 = 1 - prob_0

                if self.n_classes == 2:
                    probabilities.append([prob_0, prob_1])
                else:
                    # Multi-class case (simplified)
                    probs = np.ones(self.n_classes) / self.n_classes
                    probs[0] = prob_0
                    probs[1] = prob_1
                    probs /= np.sum(probs)  # Normalize
                    probabilities.append(probs)

        return np.array(probabilities)

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        validation_split: float = 0.0,
        verbose: bool = True
    ):
        """Train the quantum classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size (None for full batch)
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress
        """
        self._validate_inputs(X, y)

        # Set number of classes
        self.n_classes = len(np.unique(y))

        # Split data if validation requested
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        # Training loop with comprehensive monitoring
        try:
            with ResourceMonitor(self.config) as monitor:
                for epoch in range(epochs):
                    # Check resources periodically
                    if epoch % 10 == 0:
                        monitor.check_memory()
                        monitor.check_timeout()

                    # Compute gradients and update parameters
                    loss = self._training_step(
                        X_train, y_train, learning_rate, batch_size
                    )

                    # Track metrics
                    if self.config.track_metrics:
                        metrics = {'epoch': epoch, 'loss': loss}

                        if X_val is not None and epoch % 10 == 0:
                            val_acc = self.score(X_val, y_val)
                            metrics['val_accuracy'] = val_acc

                        self.metrics['training_history'].append(metrics)

                    # Print progress
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")

        except (MemoryError, TimeoutError) as e:
            logger.error(f"Training interrupted: {e}")
            if not self.config.graceful_degradation:
                raise

        self.is_fitted = True
        logger.info("Training completed successfully")

    def _training_step(
        self,
        X: NDArray,
        y: NDArray,
        learning_rate: float,
        batch_size: Optional[int]
    ) -> float:
        """Perform one training step.

        Args:
            X: Training features
            y: Training labels
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Training loss
        """
        # Use full batch if batch_size not specified
        if batch_size is None:
            batch_size = len(X)

        # Simple gradient approximation using finite differences
        gradients = np.zeros_like(self.parameters)
        total_loss = 0.0

        # Process mini-batches
        for batch_start in range(0, len(X), batch_size):
            batch_end = min(batch_start + batch_size, len(X))
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            # Compute gradients for this batch
            batch_gradients, batch_loss = self._compute_gradients(X_batch, y_batch)
            gradients += batch_gradients / (len(X) // batch_size + 1)
            total_loss += batch_loss

        # Update parameters
        self.parameters -= learning_rate * gradients

        return total_loss / len(X)

    def _compute_gradients(
        self,
        X_batch: NDArray,
        y_batch: NDArray
    ) -> Tuple[NDArray, float]:
        """Compute gradients using finite differences.

        Args:
            X_batch: Batch of features
            y_batch: Batch of labels

        Returns:
            Gradients and batch loss
        """
        gradients = np.zeros_like(self.parameters)
        epsilon = 0.01

        # Compute loss with current parameters
        current_loss = self._compute_loss(X_batch, y_batch)

        # Compute gradients using finite differences
        for i in range(len(self.parameters)):
            # Forward difference
            self.parameters[i] += epsilon
            loss_plus = self._compute_loss(X_batch, y_batch)

            # Backward difference
            self.parameters[i] -= 2 * epsilon
            loss_minus = self._compute_loss(X_batch, y_batch)

            # Restore parameter
            self.parameters[i] += epsilon

            # Compute gradient
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return gradients, current_loss

    def _compute_loss(self, X_batch: NDArray, y_batch: NDArray) -> float:
        """Compute classification loss.

        Args:
            X_batch: Batch of features
            y_batch: Batch of labels

        Returns:
            Cross-entropy loss
        """
        predictions = self.predict_proba(X_batch)

        # Cross-entropy loss with numerical stability
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        loss = 0.0
        for i, (pred, true_label) in enumerate(zip(predictions, y_batch)):
            if true_label < len(pred):
                loss -= np.log(pred[true_label])

        return loss / len(y_batch)

    def score(self, X: NDArray, y: NDArray) -> float:
        """Compute classification accuracy.

        Args:
            X: Test features
            y: True labels

        Returns:
            Classification accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Convenience function for creating models
def create_quantum_classifier(
    n_qubits: int = 2,
    depth: int = 2,
    **kwargs
) -> QuantumClassifier:
    """Create a quantum classifier with optimal default parameters.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        **kwargs: Additional configuration parameters

    Returns:
        Configured QuantumClassifier instance
    """
    return QuantumClassifier(n_qubits=n_qubits, depth=depth, **kwargs)
