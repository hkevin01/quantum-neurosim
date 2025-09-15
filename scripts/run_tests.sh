#!/bin/bash

# Test runner script for Quantum NeuroSim
# Runs comprehensive test suite with coverage reporting and performance benchmarks

set -e

# Configuration
TEST_DIR="tests"
COVERAGE_THRESHOLD=80
BENCHMARK_THRESHOLD=1.0  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo
}

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

# Check if we're in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Not in a virtual environment. Activate venv first:"
        echo "  source venv/bin/activate"
        echo
    fi
}

# Run unit tests with coverage
run_unit_tests() {
    print_header "Running Unit Tests"

    if [ ! -d "$TEST_DIR" ]; then
        print_error "Test directory not found: $TEST_DIR"
        exit 1
    fi

    # Run pytest with coverage
    print_status "Executing pytest with coverage..."
    pytest \
        --cov=src/qns \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-report=xml:coverage.xml \
        --cov-fail-under=$COVERAGE_THRESHOLD \
        --tb=short \
        --verbose \
        $TEST_DIR/unit/

    print_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"

    print_status "Testing quantum backend integration..."
    pytest \
        --tb=short \
        --verbose \
        -m "integration" \
        $TEST_DIR/integration/

    print_success "Integration tests completed"
}

# Run quantum-specific tests
run_quantum_tests() {
    print_header "Running Quantum Circuit Tests"

    print_status "Testing quantum circuits and measurements..."
    pytest \
        --tb=short \
        --verbose \
        -m "quantum" \
        $TEST_DIR/quantum/

    print_success "Quantum tests completed"
}

# Run performance benchmarks
run_benchmarks() {
    print_header "Running Performance Benchmarks"

    print_status "Benchmarking quantum algorithms..."
    pytest \
        --benchmark-only \
        --benchmark-sort=mean \
        --benchmark-compare-fail=mean:${BENCHMARK_THRESHOLD}s \
        --tb=short \
        $TEST_DIR/benchmarks/

    print_success "Benchmarks completed"
}

# Run linting and code quality checks
run_linting() {
    print_header "Running Code Quality Checks"

    print_status "Checking code style with flake8..."
    flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

    print_status "Checking imports with isort..."
    isort --check-only --diff src/ tests/

    print_status "Checking formatting with black..."
    black --check --diff src/ tests/

    print_status "Running type checks with mypy..."
    mypy src/qns/

    print_success "Code quality checks passed"
}

# Run security checks
run_security() {
    print_header "Running Security Checks"

    print_status "Scanning dependencies for vulnerabilities..."
    safety check

    print_status "Checking for security issues with bandit..."
    bandit -r src/ -f json -o bandit-report.json || true

    print_success "Security checks completed"
}

# Generate test reports
generate_reports() {
    print_header "Generating Test Reports"

    print_status "Coverage report generated: htmlcov/index.html"
    print_status "Coverage XML report: coverage.xml"

    if [ -f "bandit-report.json" ]; then
        print_status "Security report: bandit-report.json"
    fi

    # Generate summary
    echo
    echo "Test Summary:"
    echo "============="

    # Coverage percentage
    if [ -f "coverage.xml" ]; then
        COVERAGE=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
line_rate = float(root.attrib['line-rate']) * 100
print(f'{line_rate:.1f}%')
" 2>/dev/null || echo "N/A")
        echo "Coverage: $COVERAGE"
    fi

    print_success "Reports generated successfully"
}

# Main test function
run_all_tests() {
    local FAILED_TESTS=0

    # Run different test categories
    check_venv

    # Unit tests (required)
    if ! run_unit_tests; then
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_error "Unit tests failed"
    fi

    # Integration tests
    if ! run_integration_tests; then
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Integration tests failed"
    fi

    # Quantum tests
    if ! run_quantum_tests; then
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Quantum tests failed"
    fi

    # Code quality
    if ! run_linting; then
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_warning "Code quality checks failed"
    fi

    # Security (non-blocking)
    run_security || print_warning "Security checks had warnings"

    # Benchmarks (optional)
    run_benchmarks || print_warning "Performance benchmarks failed"

    # Generate reports
    generate_reports

    # Final result
    echo
    if [ $FAILED_TESTS -eq 0 ]; then
        print_success "All tests passed! ✅"
        echo "Ready for deployment 🚀"
    else
        print_error "$FAILED_TESTS test categories failed ❌"
        echo "Please fix issues before deployment"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    "unit")
        check_venv
        run_unit_tests
        ;;
    "integration")
        check_venv
        run_integration_tests
        ;;
    "quantum")
        check_venv
        run_quantum_tests
        ;;
    "lint")
        run_linting
        ;;
    "security")
        run_security
        ;;
    "bench")
        check_venv
        run_benchmarks
        ;;
    "all")
        run_all_tests
        ;;
    *)
        echo "Usage: $0 [unit|integration|quantum|lint|security|bench|all]"
        echo
        echo "Test categories:"
        echo "  unit        - Run unit tests with coverage"
        echo "  integration - Run integration tests"
        echo "  quantum     - Run quantum circuit tests"
        echo "  lint        - Run code quality checks"
        echo "  security    - Run security scans"
        echo "  bench       - Run performance benchmarks"
        echo "  all         - Run all test categories (default)"
        exit 1
        ;;
esac
