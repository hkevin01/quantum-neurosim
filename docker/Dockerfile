# Multi-stage Docker build for Quantum NeuroSim
# Optimized for both development and production use

# Base image with Python and common scientific libraries
FROM python:3.10-slim as base

# Metadata
LABEL maintainer="Quantum NeuroSim Team"
LABEL description="Quantum Neural Networks Framework"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  git \
  graphviz \
  graphviz-dev \
  libffi-dev \
  libssl-dev \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r quantum && useradd -r -g quantum quantum
RUN mkdir -p /app && chown quantum:quantum /app
WORKDIR /app

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
  vim \
  nano \
  htop \
  tree \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements*.txt ./

# Create virtual environment in container
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

# Install Jupyter and extensions
RUN pip install \
  jupyter \
  jupyterlab \
  jupyterlab-git \
  jupyterlab-lsp \
  python-lsp-server[all]

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER quantum

# Expose ports for Jupyter
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy requirements
COPY requirements.txt ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy only necessary source files
COPY src/ ./src/
COPY setup.py ./
COPY README.md ./

# Install package
RUN pip install .

# Switch to non-root user
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import qns; print('OK')" || exit 1

# Default command for production
CMD ["python", "-c", "import qns; print('Quantum NeuroSim ready!')"]

# Notebook stage for Jupyter server
FROM development as notebook

# Install additional visualization packages
RUN pip install \
  plotly \
  bokeh \
  altair \
  seaborn

# Copy example notebooks
COPY notebooks/ ./notebooks/
RUN mkdir -p ./examples ./data ./results

# Set working directory to notebooks
WORKDIR /app

# Command to start Jupyter with proper configuration
CMD ["jupyter", "lab", \
  "--ip=0.0.0.0", \
  "--port=8888", \
  "--no-browser", \
  "--allow-root", \
  "--notebook-dir=/app", \
  "--ServerApp.token=''", \
  "--ServerApp.password=''"]
