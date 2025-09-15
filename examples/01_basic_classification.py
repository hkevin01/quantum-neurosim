#!/usr/bin/env python3
"""
Basic Quantum Classification Example

This example demonstrates a simple binary classification task using
quantum neural networks with the XOR problem.

Usage:
    python 01_basic_classification.py
    python 01_basic_classification.py --n_qubits 4 --epochs 20 --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from qns.data import generate_xor_data, normalize_data, split_data
    from qns.encoders import AngleEncoder
    from qns.models import QuantumClassifier
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to install the quantum simulation package:")
    print("   pip install -e .")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantum Neural Network XOR Classification"
    )
    parser.add_argument(
        "--n_qubits", type=int, default=2,
        help="Number of qubits (default: 2)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of quantum layers (default: 2)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of training samples (default: 100)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="Noise level for data generation (default: 0.1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    return parser.parse_args()


def print_header():
    """Print example header."""
    print("🚀 Quantum Neural Network - XOR Classification")
    print("=" * 50)


def generate_and_prepare_data(
    n_samples, noise_level, test_size, seed, verbose
):
    """Generate and prepare the XOR dataset."""
    if verbose:
        print("📊 Generating XOR dataset...")
        print(f"   Samples: {n_samples}")
        print(f"   Noise level: {noise_level}")
        print(f"   Test split: {test_size}")

    # Generate XOR data
    X, y = generate_xor_data(
        n_samples=n_samples,
        noise_level=noise_level,
        random_seed=seed
    )

    # Normalize features
    X = normalize_data(X, method='minmax')

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_seed=seed
    )

    if verbose:
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Feature range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Label distribution: {np.bincount(y)}")

    return X_train, X_test, y_train, y_test


def create_quantum_model(n_qubits, n_layers, learning_rate, verbose):
    """Create and configure the quantum classifier."""
    if verbose:
        print("⚛️  Creating quantum classifier...")
        print(f"   Qubits: {n_qubits}")
        print(f"   Layers: {n_layers}")
        print(f"   Learning rate: {learning_rate}")

    # Create encoder
    encoder = AngleEncoder(n_qubits=n_qubits, normalize=True)

    # Create quantum classifier
    classifier = QuantumClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        learning_rate=learning_rate
    )

    if verbose:
        print(f"   Encoder: {encoder.__class__.__name__}")
        print(f"   Model: {classifier.__class__.__name__}")

    return classifier, encoder


def train_model(classifier, X_train, y_train, epochs, verbose):
    """Train the quantum classifier."""
    if verbose:
        print("🎯 Training model...")
        print(f"   Epochs: {epochs}")
        print(f"   Training samples: {len(X_train)}")

    start_time = time.time()

    try:
        # Train the model
        history = classifier.fit(
            X_train, y_train,
            epochs=epochs,
            verbose=verbose
        )

        training_time = time.time() - start_time

        if verbose:
            print(f"   Training completed in {training_time:.2f} seconds")
            if hasattr(classifier, 'training_history'):
                print(f"   Final loss: {classifier.training_history[-1]:.4f}")

        return history

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def evaluate_model(classifier, X_train, y_train, X_test, y_test, verbose):
    """Evaluate the trained model."""
    if verbose:
        print("📈 Evaluating model...")

    try:
        # Training accuracy
        train_accuracy = classifier.score(X_train, y_train)

        # Test accuracy
        test_predictions = classifier.predict(X_test)
        test_accuracy = classifier.score(X_test, y_test)

        if verbose:
            print(f"   Training accuracy: {train_accuracy:.4f}")
            print(f"   Test accuracy: {test_accuracy:.4f}")

            # Show some predictions
            print("   Sample predictions (first 5):")
            for i in range(min(5, len(X_test))):
                actual = y_test[i]
                predicted = test_predictions[i]
                correct = "✓" if actual == predicted else "✗"
                print(f"     {correct} Actual: {actual}, "
                      f"Predicted: {predicted}")

        return train_accuracy, test_accuracy

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None, None


def save_results(classifier, results_dir="results"):
    """Save the trained model and results."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    try:
        model_path = results_path / "xor_quantum_classifier.pkl"
        classifier.save(str(model_path))
        print(f"💾 Model saved to: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")
        return None


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)

    # Print header
    print_header()

    try:
        # Generate and prepare data
        X_train, X_test, y_train, y_test = generate_and_prepare_data(
            args.samples, args.noise, args.test_size, args.seed, args.verbose
        )

        # Create quantum model
        classifier, encoder = create_quantum_model(
            args.n_qubits, args.n_layers, args.learning_rate, args.verbose
        )

        # Train model
        history = train_model(
            classifier, X_train, y_train, args.epochs, args.verbose
        )

        if history is None:
            print("❌ Training failed, cannot continue")
            return 1

        # Evaluate model
        train_acc, test_acc = evaluate_model(
            classifier, X_train, y_train, X_test, y_test, args.verbose
        )

        if train_acc is None:
            print("❌ Evaluation failed")
            return 1

        # Save results
        if args.verbose:
            save_results(classifier)

        # Print summary
        print("\n📋 Summary")
        print("-" * 20)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Model Parameters: {args.n_qubits} qubits, "
              f"{args.n_layers} layers")

        if test_acc > 0.8:
            print("🎉 Great performance! The model learned the "
                  "XOR function well.")
        elif test_acc > 0.6:
            print("👍 Decent performance. Consider tuning hyperparameters.")
        else:
            print("🤔 Low performance. Try increasing epochs or "
                  "adjusting parameters.")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
