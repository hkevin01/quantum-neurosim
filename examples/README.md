# Quantum NeuroSim Examples

This directory contains example scripts and notebooks demonstrating the capabilities of the quantum neural simulation framework.

## Examples Overview

### Basic Examples

- **01_basic_classification.py**: Simple binary classification with XOR data
- **02_encoder_comparison.py**: Comparison of different quantum encoding strategies
- **03_model_training.py**: Complete model training workflow
- **04_data_generation.py**: Using built-in data generators

### Advanced Examples

- **05_noise_models.py**: Training with quantum noise
- **06_hybrid_networks.py**: Hybrid classical-quantum networks
- **07_optimization.py**: Advanced optimization techniques
- **08_benchmarking.py**: Performance benchmarking

### Notebooks

- **tutorial_01_getting_started.ipynb**: Introduction to quantum neural networks
- **tutorial_02_encoding_strategies.ipynb**: Deep dive into quantum encoding
- **tutorial_03_training_techniques.ipynb**: Advanced training methods
- **tutorial_04_real_hardware.ipynb**: Running on real quantum devices

## Running Examples

### Python Scripts

```bash
# Run basic classification example
python examples/01_basic_classification.py

# Run with different parameters
python examples/01_basic_classification.py --n_qubits 4 --epochs 50
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab examples/

# Or start notebook server
jupyter notebook examples/
```

### Docker Environment

```bash
# Run in Docker container with all dependencies
docker-compose -f docker/docker-compose.yml up quantum-neurosim-notebooks -d

# Access at http://localhost:8889
```

## Example Requirements

Most examples require the base quantum simulation framework. Some advanced examples may require additional dependencies:

- **Real hardware examples**: IBM Quantum account and Qiskit
- **Advanced optimization**: Additional optimization packages
- **Visualization examples**: Matplotlib, Plotly, or other plotting libraries

## Data Requirements

Examples use built-in data generators by default. Some examples may download external datasets:

- **Quantum datasets**: Generated synthetically
- **Classical benchmarks**: Iris, Wine, Breast Cancer (via scikit-learn)
- **Time series**: Generated or from public datasets

## Configuration

Examples can be configured via:

- **Command line arguments**: Most scripts accept parameters
- **Configuration files**: JSON/YAML config files in `examples/configs/`
- **Environment variables**: Set QNS_* variables for global settings

## Output and Results

Examples generate output in:

- **Console**: Basic results and progress
- **Files**: Trained models, plots, and data in `results/` directory
- **Logs**: Detailed execution logs for debugging

## Contributing Examples

To contribute new examples:

1. Follow the naming convention: `NN_descriptive_name.py`
2. Include docstrings and comments
3. Add command line argument parsing
4. Include example usage in the docstring
5. Test with different parameter combinations

## Troubleshooting

### Common Issues

**Import errors**: Ensure the quantum simulation package is installed
```bash
pip install -e .
```

**Memory errors**: Reduce the number of qubits or samples for large examples

**Slow execution**: Use simulators instead of real quantum hardware for development

**Missing dependencies**: Install additional requirements
```bash
pip install -r requirements-examples.txt
```

### Getting Help

- Check the main README for installation instructions
- Review the API documentation
- Open an issue on GitHub for bugs or feature requests
- Join the community discussions for questions
