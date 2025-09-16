# 🐳 Docker Integration Guide

This document provides comprehensive guidance on containerizing your quantum neural simulation environment using Docker. Based on advanced Docker practices specifically tailored for quantum computing development.

## 🚀 Quick Start

```bash
# Clone and enter the project
git clone <repository-url>
cd quantum-neurosim

# Start the quantum development environment
chmod +x scripts/*.sh
./scripts/start.sh

# Access Jupyter Lab
open http://localhost:8888
```

## 📁 Docker Architecture

```
docker/
├── Dockerfile.cpu-quantum          # CPU-optimized quantum environment
├── Dockerfile.gpu-quantum          # GPU-accelerated quantum environment
├── docker-compose.advanced.yml     # Advanced orchestration
├── requirements-quantum.txt        # Pinned quantum library versions
└── ...

config/
├── redis.conf                      # Redis cache configuration
└── ...

monitoring/
├── prometheus.yml                   # Metrics collection
├── rules/
│   └── quantum_alerts.yml          # Alert rules
└── ...

scripts/
├── start.sh                        # Environment startup
├── stop.sh                         # Environment shutdown
└── backup.sh                       # Data backup utility
```

## 🏗️ Container Types

### 1. CPU-Optimized Development (Default)

- **Image**: `Dockerfile.cpu-quantum`
- **Purpose**: General quantum development, small to medium circuits
- **Resources**: Standard CPU, 4-8GB RAM
- **Ports**: 8888 (Jupyter), 8080 (Dashboard)

```bash
docker-compose up quantum-dev
```

### 2. GPU-Accelerated Environment

- **Image**: `Dockerfile.gpu-quantum`
- **Purpose**: Large quantum simulations, ML training
- **Resources**: NVIDIA GPU, 8-16GB RAM, CUDA 12.3.2
- **Ports**: 8889 (Jupyter)

```bash
./scripts/start.sh --gpu
```

### 3. Production Service

- **Purpose**: API server, production workloads
- **Features**: Minimal image, health checks, monitoring
- **Ports**: 8000 (API)

```bash
docker-compose --profile production up
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Copy template and customize
cp .env.template .env
nano .env

# Key variables:
QISKIT_IBM_TOKEN=your_token
AWS_ACCESS_KEY_ID=your_key
CUDA_VISIBLE_DEVICES=0
```

### Cloud Provider Authentication

#### IBM Quantum

```bash
# Method 1: Environment variables
export QISKIT_IBM_TOKEN="your_token"
export QISKIT_IBM_CHANNEL="ibm_quantum"


# Method 2: Mount credentials directory
-v ~/.qiskit:/home/quser/.qiskit:ro
```

#### AWS Braket

```bash
# Method 1: Mount AWS credentials
-v ~/.aws:/home/quser/.aws:ro


# Method 2: Environment variables
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

#### Rigetti QCS

```bash

# Mount QCS configuration
-v ~/.qcs:/home/quser/.qcs:ro
```

## 📊 Monitoring & Observability

### Prometheus Metrics (Port 9090)

- Container resource usage
- Quantum circuit execution times
- Database connection health
- GPU utilization (if available)

### Grafana Dashboards (Port 3000)

- Real-time performance monitoring
- Historical execution trends
- Resource utilization graphs
- Alert management

```bash
# Start monitoring stack
./scripts/start.sh --monitoring
```

## 🔄 Service Orchestration

### Core Services

- **quantum-dev**: Main development environment

- **quantum-db**: PostgreSQL for experiment tracking
- **quantum-redis**: Results caching

### Optional Services

- **quantum-gpu**: GPU-accelerated environment
- **quantum-monitor**: Prometheus monitoring
- **quantum-grafana**: Visualization dashboards
- **quantum-proxy**: Nginx reverse proxy

### Service Profiles

```bash
# Development only
docker-compose up

# Include GPU support

docker-compose --profile gpu up

# Production deployment
docker-compose --profile production up

# Full monitoring stack
docker-compose --profile monitoring up

```

## 💾 Data Management

### Persistent Volumes

- **quantum-db-data**: PostgreSQL database
- **quantum-redis-data**: Redis cache
- **quantum-cache**: Circuit compilation cache
- **quantum-gpu-cache**: GPU-specific cache
- **quantum-metrics**: Prometheus data

### Backup Strategy

```bash
# Manual backup

./scripts/backup.sh

# Automated backups (add to crontab)
0 2 * * * /path/to/quantum-neurosim/scripts/backup.sh
```

## 🚀 Performance Optimization

### CPU Environment

- **Memory**: 4-8GB minimum, 16GB recommended for >12 qubits
- **CPU**: Multi-core processor, Intel/AMD x64
- **Storage**: SSD recommended for database operations

### GPU Environment

- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+
- **Memory**: 8GB+ GPU memory for large circuits
- **Drivers**: NVIDIA Driver 470.57.02+, CUDA 12.3.2

- **Libraries**: cuQuantum for additional acceleration

### Container Overhead

- Typical overhead: 5-15% for quantum workloads
- Memory overhead: ~200MB base container
- Network latency: <2ms additional for cloud connections

## 🔐 Security Best Practices

### Container Security

- Non-root user execution (`quser`)
- Read-only credential mounts
- Network isolation via custom bridge
- Resource limits and health checks

### Credential Management

```bash
# Store credentials securely

echo "QISKIT_IBM_TOKEN=..." >> .env
chmod 600 .env

# Use Docker secrets in production
docker secret create ibm_token /path/to/token
```

### Network Security

```bash
# Production deployment with SSL
docker-compose --profile production up

# Nginx proxy with SSL termination on port 443
```

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start

```bash

# Check Docker daemon
docker info

# Rebuild images
./scripts/start.sh --rebuild

# Check logs
docker-compose logs quantum-dev
```

#### GPU Not Detected

```bash
# Verify NVIDIA runtime
nvidia-smi
docker run --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support

docker run --gpus all --rm nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

#### Memory Issues

```bash
# Check container memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB+
```

#### Cloud Connection Failures

```bash
# Test credentials
docker exec -it quantum-dev python -c "from qiskit_ibm_runtime import QiskitRuntimeService; print(QiskitRuntimeService().backends())"

# Check network connectivity
docker exec -it quantum-dev curl -I https://quantum-computing.ibm.com
```

### Debug Mode

```bash

# Start with debug logging
export QUANTUM_LOG_LEVEL=DEBUG
./scripts/start.sh

# Interactive shell access
docker exec -it quantum-dev bash
```

## 📈 Production Deployment

### Docker Compose Production

```yaml
# Minimal production setup
services:
  quantum-prod:
    image: quantum-neurosim:latest
    environment:
      - QN_ENV=production
      - QUANTUM_LOG_LEVEL=WARNING
    volumes:
      - production-results:/app/results
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment

metadata:
  name: quantum-neurosim
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-neurosim
  template:
    metadata:
      labels:
        app: quantum-neurosim

    spec:
      containers:
      - name: quantum-app
        image: quantum-neurosim:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:

            memory: "8Gi"
            cpu: "4"
```

## 🔗 Integration Examples

### Jupyter Notebook Integration

```python
# In quantum_docker_tutorial.ipynb
import os<https://docs.docker.com/>
print(f"Running in container: {os.path.exists('/.dockerenv')}")

# Use cached quantum results
from qns.data.cache import QuantumResultCache
cache = QuantumResultCache(redis_url="redis://quantum-redis:6379")
```

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Test Quantum Framework
  run: |
    docker-compose -f docker/docker-compose.advanced.yml build
    docker-compose -f docker/docker-compose.advanced.yml run --rm quantum-dev pytest
```

### Cloud Deployment

```bash
# AWS ECS/Fargate deployment
aws ecs create-service \
  --cluster quantum-cluster \
  --service-name quantum-neurosim \
  --task-definition quantum-neurosim:1
```

## 📚 Additional Resources

- **Docker Documentation**: <https://docs.docker.com/>
- **Quantum Computing with Docker**: `/notebooks/quantum_docker_tutorial.ipynb`
- **Performance Benchmarking**: Check monitoring dashboards
- **Community Support**: GitHub Issues and Discussions

## 🏁 Next Steps

1. **Start Development**: `./scripts/start.sh`
2. **Open Tutorial**: Navigate to `notebooks/quantum_docker_tutorial.ipynb`
3. **Run Examples**: Execute `examples/01_basic_classification.py`
4. **Configure Monitoring**: `./scripts/start.sh --monitoring`
5. **Setup Cloud Access**: Update `.env` with your credentials
6. **Enable GPU**: `./scripts/start.sh --gpu` (if available)

---

*Docker serves as the "quantum development OS" - providing consistent, reproducible environments for quantum computing development, simulation, and deployment.*
