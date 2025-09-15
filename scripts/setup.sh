#!/bin/bash

# Setup script for Quantum NeuroSim development environment
# This script automates the setup process for new developers

set -e  # Exit on any error

echo "🚀 Setting up Quantum NeuroSim development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"

        # Check if version is 3.9 or higher
        REQUIRED_VERSION="3.9"
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or higher."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi

    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"

    # Upgrade pip
    pip install --upgrade pip
    print_success "Pip upgraded to latest version"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."

    # Install core dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Core dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi

    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    else
        print_warning "requirements-dev.txt not found, skipping dev dependencies"
    fi

    # Install package in development mode
    pip install -e .
    print_success "Package installed in development mode"
}

# Setup pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "Pre-commit not available, skipping hook setup"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."

    # Create data directories
    mkdir -p data/{raw,processed,experiments}
    mkdir -p results/{models,plots,logs}
    mkdir -p experiments/configs

    # Create .gitkeep files
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch data/experiments/.gitkeep
    touch results/models/.gitkeep
    touch results/plots/.gitkeep
    touch results/logs/.gitkeep

    print_success "Project directories created"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Test import
    python3 -c "
import qns
print(f'✅ Quantum NeuroSim v{qns.__version__} imported successfully')

# Test quantum backends
try:
    import qiskit
    print('✅ Qiskit available')
except ImportError:
    print('❌ Qiskit not available')

try:
    import pennylane
    print('✅ PennyLane available')
except ImportError:
    print('❌ PennyLane not available')

# Test core functionality
from qns.data import generate_xor_data
from qns.models import QuantumClassifier

X, y = generate_xor_data(n_samples=10)
print('✅ Data generation working')

model = QuantumClassifier(n_qubits=2, depth=1)
print('✅ Model creation working')

print('\n🎉 Installation verification complete!')
"

    print_success "Installation verified successfully"
}

# Run setup steps
main() {
    echo "============================================"
    echo "  Quantum NeuroSim Development Setup"
    echo "============================================"
    echo

    check_python
    create_venv
    install_dependencies
    setup_precommit
    create_directories
    verify_installation

    echo
    echo "============================================"
    echo "          Setup Complete! 🎉"
    echo "============================================"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run tests: pytest tests/"
    echo "3. Start Jupyter: jupyter lab"
    echo "4. Check the documentation in docs/"
    echo
    print_success "Happy quantum machine learning! 🧠⚛️"
}

# Run main function
main "$@"
