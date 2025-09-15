#!/bin/bash

# Quick test runner script for development
# This script runs a subset of tests to verify the framework

set -e

echo "🧪 Running Quantum NeuroSim Tests..."
echo "=================================="

# Change to project directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov pytest-mock
fi

echo "🔍 Discovering tests..."
python -m pytest --collect-only tests/ | grep "test session starts" || echo "Test collection complete"

echo ""
echo "🚀 Running unit tests..."
python -m pytest tests/unit/ -v --tb=short -x

echo ""
echo "✅ Quick tests completed!"

# Check if any tests failed
if [ $? -eq 0 ]; then
    echo "🎉 All tests passed!"
else
    echo "❌ Some tests failed. Check output above."
    exit 1
fi
