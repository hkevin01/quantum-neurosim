"""Quantum NeuroSim - A comprehensive framework for quantum neural networks.

This package provides tools and algorithms for implementing, training, and
analyzing quantum neural networks using various quantum computing frameworks.

Key Components:
- Quantum circuit encoders for classical-to-quantum data transformation
- Variational quantum circuits for parameterized quantum neural networks
- Training algorithms optimized for quantum hardware
- Measurement strategies and observable construction
- Error mitigation and noise handling utilities
- Visualization and analysis tools

Supported Backends:
- Qiskit (IBM Quantum)
- PennyLane (Xanadu)
- Cirq (Google)
- Amazon Braket
- PyQuil (Rigetti)

Example:
    >>> from qns.models import QuantumClassifier
    >>> from qns.data import load_xor_dataset
    >>>
    >>> # Load data and create classifier
    >>> X, y = load_xor_dataset()
    >>> classifier = QuantumClassifier(n_qubits=2, depth=2)
    >>>
    >>> # Train the model
    >>> classifier.fit(X, y, epochs=100, learning_rate=0.1)
    >>>
    >>> # Make predictions
    >>> predictions = classifier.predict(X)
"""

import logging

__version__ = "0.1.0"
__author__ = "Quantum NeuroSim Team"
__email__ = "quantum-neurosim@example.com"

# Version information
version_info = tuple(map(int, __version__.split('.')))

# Configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())
