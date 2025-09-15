# Contributing to Quantum NeuroSim

Thank you for your interest in contributing to Quantum NeuroSim! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## 🤝 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports**: Help us identify and fix issues
- **💡 Feature Requests**: Suggest new functionality
- **📚 Documentation**: Improve or add documentation
- **🔬 Research**: Contribute quantum algorithms or neural network architectures
- **🧪 Testing**: Add test cases and improve coverage
- **🎨 Examples**: Create tutorials and example notebooks

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `develop`
4. **Make your changes** following our coding standards
5. **Test your changes** thoroughly
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- Git
- Docker (optional, for containerized development)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-neurosim.git
cd quantum-neurosim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Docker Setup

```bash
# Build development container
docker build -f docker/Dockerfile.dev -t quantum-neurosim:dev .

# Run development environment
docker run -it --rm -v $(pwd):/workspace quantum-neurosim:dev
```

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Use `isort` for import organization
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings required

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

### Naming Conventions

```python
# Classes: PascalCase
class QuantumNeuron:
    pass

# Functions and variables: snake_case
def create_quantum_circuit():
    circuit_depth = 10

# Constants: UPPER_SNAKE_CASE
DEFAULT_SHOTS = 1024
MAX_QUBITS = 127

# Private methods: _leading_underscore
def _internal_helper():
    pass

# Quantum-specific naming
def encode_classical_data():      # Not: encode_data()
def measure_expectation_value():  # Not: measure()
def apply_variational_layer():   # Not: apply_layer()
```

### Documentation Standards

```python
def train_quantum_classifier(
    circuit: QuantumCircuit,
    training_data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01
) -> Tuple[np.ndarray, List[float]]:
    """Train a quantum classifier using gradient descent.

    This function implements a hybrid quantum-classical training loop
    for parameterized quantum circuits acting as classifiers.

    Args:
        circuit: The parameterized quantum circuit to train
        training_data: Input data of shape (n_samples, n_features)
        labels: Target labels of shape (n_samples,)
        epochs: Number of training epochs (default: 100)
        learning_rate: Step size for parameter updates (default: 0.01)

    Returns:
        A tuple containing:
            - Trained parameters as numpy array
            - List of loss values during training

    Raises:
        ValueError: If training_data and labels have incompatible shapes
        QuantumError: If circuit execution fails

    Example:
        >>> circuit = create_variational_classifier(n_qubits=2)
        >>> X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        >>> y = np.array([1, 1, 0, 0])
        >>> params, losses = train_quantum_classifier(circuit, X, y)
    """
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_encoders.py
│   ├── test_circuits.py
│   └── test_optimizers.py
├── integration/          # Integration tests
│   ├── test_training_loop.py
│   └── test_experiments.py
├── quantum/              # Quantum-specific tests
│   ├── test_simulators.py
│   └── test_hardware.py
└── conftest.py          # Test configuration and fixtures
```

### Writing Tests

```python
import pytest
import numpy as np
from qns.models import QuantumClassifier

class TestQuantumClassifier:
    """Test suite for QuantumClassifier."""

    @pytest.fixture
    def sample_data(self):
        """Create sample XOR dataset."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        return X, y

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = QuantumClassifier(n_qubits=2, depth=2)
        assert classifier.n_qubits == 2
        assert classifier.depth == 2
        assert len(classifier.parameters) == 8  # 2 * 2 * 2

    def test_training_convergence(self, sample_data):
        """Test that training reduces loss."""
        X, y = sample_data
        classifier = QuantumClassifier(n_qubits=2, depth=2)

        initial_loss = classifier.compute_loss(X, y)
        classifier.train(X, y, epochs=50, learning_rate=0.1)
        final_loss = classifier.compute_loss(X, y)

        assert final_loss < initial_loss

    @pytest.mark.parametrize("n_qubits,depth", [
        (2, 1), (2, 2), (3, 1), (4, 2)
    ])
    def test_different_architectures(self, n_qubits, depth):
        """Test various circuit architectures."""
        classifier = QuantumClassifier(n_qubits=n_qubits, depth=depth)
        expected_params = n_qubits * depth * 2
        assert len(classifier.parameters) == expected_params

    @pytest.mark.slow
    def test_hardware_compatibility(self):
        """Test compatibility with quantum hardware."""
        # This test requires actual quantum backend access
        pytest.skip("Hardware tests require backend configuration")
```

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Slow tests (>5 seconds)
@pytest.mark.hardware      # Requires quantum hardware
@pytest.mark.gpu           # Requires GPU
```

## 📤 Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/quantum-hopfield-network
   ```

2. **Make your changes** with clear, atomic commits:
   ```bash
   git commit -m "feat: implement quantum Hopfield network

   - Add QHopfieldNetwork class with VQE-based energy minimization
   - Implement pattern storage using Hebbian weights
   - Add noise robustness testing
   - Update documentation with usage examples

   Closes #123"
   ```

3. **Update documentation** and tests

4. **Run the test suite**:
   ```bash
   pytest tests/
   black src/ tests/
   isort src/ tests/
   mypy src/
   ```

5. **Push and create pull request**

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```bash
feat(quantum): add QAOA optimizer for combinatorial problems
fix(training): resolve gradient explosion in deep circuits
docs(api): update quantum encoder documentation
test(integration): add end-to-end XOR classification test
```

## 🔍 Review Process

### Review Criteria

Pull requests are reviewed based on:

1. **Correctness**: Code works as intended
2. **Quantum Validity**: Quantum algorithms are theoretically sound
3. **Performance**: Efficient use of quantum resources
4. **Testing**: Adequate test coverage (>80%)
5. **Documentation**: Clear documentation and examples
6. **Style**: Follows project coding standards

### Review Timeline

- **Initial review**: Within 48 hours
- **Follow-up reviews**: Within 24 hours of updates
- **Final approval**: Requires 2 approvals from maintainers

### Reviewer Guidelines

Reviewers should check:

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] Quantum circuits are efficient
- [ ] No breaking changes (or properly documented)
- [ ] Performance implications considered
- [ ] Security considerations addressed

## 🎯 Development Focus Areas

### High Priority

1. **Quantum Algorithm Implementation**
   - New variational quantum algorithms
   - Improved gradient computation methods
   - Hardware-efficient circuit designs

2. **Neural Network Architectures**
   - Quantum convolutional networks
   - Attention mechanisms for quantum circuits
   - Hybrid classical-quantum architectures

3. **Performance Optimization**
   - Circuit depth reduction
   - Shot count optimization
   - Batch processing for training

### Research Contributions

We especially welcome contributions in:

- Novel quantum machine learning algorithms
- Theoretical analysis of quantum neural networks
- Benchmarking studies against classical methods
- Hardware noise characterization and mitigation
- Quantum advantage demonstrations

## 💬 Communication

### Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Code-specific discussions

### Getting Help

- Check existing documentation and examples
- Search through GitHub issues
- Create a new issue with the "question" label
- Join our community discussions

## 📜 License

By contributing to Quantum NeuroSim, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to Quantum NeuroSim! Together, we're advancing the frontier of quantum machine learning. 🚀
