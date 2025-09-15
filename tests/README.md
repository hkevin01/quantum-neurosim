# Tests for Quantum NeuroSim

This directory contains comprehensive test suites for the quantum neural simulation framework.

## Test Structure

```text
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Pytest fixtures and configuration
├── unit/                      # Unit tests for individual components
│   ├── __init__.py
│   ├── test_encoders.py       # Test quantum data encoders
│   ├── test_models.py         # Test quantum neural network models
│   └── test_data.py           # Test data utilities
├── integration/               # Integration tests for component interactions
│   ├── __init__.py
│   ├── test_training.py       # Test end-to-end training workflows
│   ├── test_backends.py       # Test quantum backend integrations
│   └── test_persistence.py    # Test model saving/loading
├── quantum/                   # Quantum-specific tests requiring simulators
│   ├── __init__.py
│   ├── test_circuits.py       # Test quantum circuit construction
│   ├── test_execution.py      # Test quantum circuit execution
│   └── test_noise.py          # Test noise models and error mitigation
├── benchmarks/                # Performance and benchmark tests
│   ├── __init__.py
│   ├── test_performance.py    # Performance regression tests
│   └── test_scalability.py    # Scalability tests for large datasets
└── fixtures/                  # Test data and fixtures
    ├── sample_data.npz        # Sample datasets for testing
    └── reference_models/      # Reference quantum models
```

## Running Tests

### All Tests

```bash
# Run full test suite with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run with verbose output
pytest tests/ -v
```

### Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Quantum tests (requires quantum simulators)
pytest tests/quantum/

# Benchmark tests
pytest tests/benchmarks/
```

### Specific Test Files

```bash
# Test specific component
pytest tests/unit/test_encoders.py

# Test with specific markers
pytest tests/ -m "not slow"
pytest tests/ -m "quantum"
```

## Test Markers

- `@pytest.mark.slow`: Long-running tests (>10 seconds)
- `@pytest.mark.quantum`: Tests requiring quantum simulators
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.benchmark`: Performance benchmark tests
- `@pytest.mark.gpu`: Tests requiring GPU acceleration
- `@pytest.mark.hardware`: Tests requiring real quantum hardware

## Test Configuration

Tests are configured via `conftest.py` with the following features:

- Automatic quantum backend detection
- Test data fixtures and generators
- Performance monitoring and reporting
- Resource cleanup and isolation
- Mock quantum hardware when unavailable

## Coverage Requirements

- Minimum 85% code coverage for all modules
- 100% coverage required for critical quantum operations
- Documentation coverage tracking for public APIs

## Continuous Integration

Tests run automatically on:

- Every pull request
- Merge to main branch
- Nightly builds with extended test matrix
- Release candidate validation

## Writing New Tests

### Unit Test Template

```python
import pytest
from qns.encoders import AngleEncoder

class TestAngleEncoder:
    def test_initialization(self):
        encoder = AngleEncoder(n_qubits=4)
        assert encoder.n_qubits == 4

    def test_encoding(self):
        encoder = AngleEncoder(n_qubits=2)
        data = [0.5, 0.8]
        circuit = encoder.encode(data)
        assert circuit is not None
```

### Integration Test Template

```python
import pytest
from qns.models import QuantumClassifier
from qns.data import generate_xor_data

@pytest.mark.integration
class TestQuantumTraining:
    def test_xor_training(self, quantum_backend):
        X, y = generate_xor_data(100)
        model = QuantumClassifier(
            n_qubits=2,
            backend=quantum_backend
        )
        model.fit(X, y, epochs=10)
        accuracy = model.score(X, y)
        assert accuracy > 0.8
```

## Test Data

Test fixtures include:

- **XOR Dataset**: 2D binary classification
- **Iris Dataset**: Multi-class classification
- **Synthetic Quantum**: Quantum state datasets
- **Time Series**: Temporal pattern data
- **Noisy Data**: Datasets with various noise models

## Performance Baselines

Benchmark tests validate:

- Training time < 30 seconds for small datasets
- Memory usage < 1GB for standard models
- Circuit depth optimization within 20% of theoretical minimum
- Gradient computation accuracy within 1e-6 tolerance
