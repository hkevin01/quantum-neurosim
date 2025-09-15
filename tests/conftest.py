"""
Pytest configuration and fixtures for quantum neural simulation tests.

This module provides common fixtures, test utilities, and configuration
for the quantum neural simulation test suite.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator, List, Optional
from unittest.mock import Mock

import numpy as np
import pytest

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path setup
try:
    from qiskit import Aer
    from qiskit.providers.fake_provider import FakeVigo
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False


# Pytest markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests as requiring quantum simulators"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU acceleration"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests as requiring real quantum hardware"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]  # noqa: ARG001
) -> None:
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests in quantum/ directory as quantum tests
        if "quantum/" in str(item.fspath):
            item.add_marker(pytest.mark.quantum)

        # Mark tests in integration/ directory as integration tests
        if "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in benchmarks/ directory as benchmark tests
        if "benchmarks/" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)

        # Mark slow tests based on name patterns
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# Test data fixtures
@pytest.fixture
def sample_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample XOR dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


@pytest.fixture
def sample_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample multi-class classification dataset."""
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Generate sample time series data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    noise = 0.1 * np.random.randn(100)
    return np.sin(t) + 0.5 * np.sin(3 * t) + noise


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Quantum backend fixtures
@pytest.fixture
def qiskit_simulator():
    """Provide Qiskit Aer simulator if available."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")

    return Aer.get_backend('qasm_simulator')


@pytest.fixture
def qiskit_statevector_simulator():
    """Provide Qiskit statevector simulator if available."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")

    return Aer.get_backend('statevector_simulator')


@pytest.fixture
def qiskit_fake_device():
    """Provide Qiskit fake device for noise testing."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")

    return FakeVigo()


@pytest.fixture
def pennylane_device():
    """Provide PennyLane device if available."""
    if not PENNYLANE_AVAILABLE:
        pytest.skip("PennyLane not available")

    return qml.device('default.qubit', wires=4)


@pytest.fixture
def cirq_simulator():
    """Provide Cirq simulator if available."""
    if not CIRQ_AVAILABLE:
        pytest.skip("Cirq not available")

    return cirq.Simulator()


@pytest.fixture(params=['qiskit', 'pennylane', 'cirq'])
def quantum_backend(request):
    """Parametrized fixture providing different quantum backends."""
    backend_type = request.param

    if backend_type == 'qiskit':
        if not QISKIT_AVAILABLE:
            pytest.skip("Qiskit not available")
        return Aer.get_backend('qasm_simulator')

    elif backend_type == 'pennylane':
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        return qml.device('default.qubit', wires=4)

    elif backend_type == 'cirq':
        if not CIRQ_AVAILABLE:
            pytest.skip("Cirq not available")
        return cirq.Simulator()

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Mock fixtures for testing without quantum libraries
@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing without real simulators."""
    mock_backend = Mock()
    mock_backend.name = 'mock_simulator'
    mock_backend.configuration.return_value.n_qubits = 20
    mock_backend.run.return_value.result.return_value.get_counts.return_value = {
        '00': 512, '11': 512
    }
    return mock_backend


@pytest.fixture
def mock_quantum_circuit():
    """Mock quantum circuit for testing."""
    mock_circuit = Mock()
    mock_circuit.num_qubits = 4
    mock_circuit.depth.return_value = 10
    mock_circuit.size.return_value = 20
    return mock_circuit


# Environment fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Disable warnings for cleaner test output
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set test-specific configuration
    os.environ['QNS_LOG_LEVEL'] = 'ERROR'
    os.environ['QNS_CACHE_DISABLED'] = 'true'
    os.environ['QNS_TEST_MODE'] = 'true'

    yield

    # Cleanup
    for key in ['QNS_LOG_LEVEL', 'QNS_CACHE_DISABLED', 'QNS_TEST_MODE']:
        os.environ.pop(key, None)


@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import time

    import psutil

    # Record start metrics
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    # Record end metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    duration = end_time - start_time
    memory_delta = end_memory - start_memory

    # Warn about slow tests or memory leaks
    if duration > 10.0:
        pytest.warns(UserWarning, match=f"Slow test: {duration:.2f}s")

    if memory_delta > 100:  # MB
        pytest.warns(UserWarning, match=f"Memory usage: {memory_delta:.2f}MB")


# Utility functions for tests
def assert_quantum_circuit_valid(circuit: Any, expected_qubits: int) -> None:
    """Assert that a quantum circuit is valid."""
    assert circuit is not None, "Circuit should not be None"

    if hasattr(circuit, 'num_qubits'):
        assert circuit.num_qubits == expected_qubits, (
            f"Expected {expected_qubits} qubits, got {circuit.num_qubits}"
        )
    elif hasattr(circuit, 'n_qubits'):
        assert circuit.n_qubits == expected_qubits, (
            f"Expected {expected_qubits} qubits, got {circuit.n_qubits}"
        )


def assert_array_properties(
    array: np.ndarray,
    expected_shape: tuple,
    expected_dtype: Optional[type] = None,
    expected_range: Optional[tuple] = None
) -> None:
    """Assert properties of numpy arrays."""
    assert isinstance(array, np.ndarray), "Input should be numpy array"
    assert array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {array.shape}"
    )

    if expected_dtype:
        assert array.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {array.dtype}"
        )

    if expected_range:
        min_val, max_val = expected_range
        assert np.all(array >= min_val) and np.all(array <= max_val), (
            f"Array values should be in range [{min_val}, {max_val}], "
            f"got [{array.min()}, {array.max()}]"
        )


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    try:
        import cupy
        _ = cupy.cuda.Device()
    except (ImportError, RuntimeError):
        pytest.skip("GPU not available")


def skip_if_no_internet():
    """Skip test if no internet connection available."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        pytest.skip("Internet connection required")


# Test configuration
pytest_plugins = ['pytest_benchmark', 'pytest_xdist', 'pytest_mock']
