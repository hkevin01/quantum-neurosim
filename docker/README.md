# Docker configurations and deployment files

## Quick Start

### Development Environment

```bash
# Start development environment with Jupyter Lab
docker-compose -f docker/docker-compose.yml up quantum-neurosim-dev -d

# Access Jupyter Lab at: http://localhost:8888
# Password will be shown in container logs
```

### Notebook Server

```bash
# Start notebook server for demos
docker-compose -f docker/docker-compose.yml up quantum-neurosim-notebooks -d

# Access notebooks at: http://localhost:8889
```

### Full Stack

```bash
# Start all services including database and monitoring
docker-compose -f docker/docker-compose.yml up -d
```

## Services Overview

### Core Services

- **quantum-neurosim-dev**: Development environment with Jupyter Lab (port 8888)
- **quantum-neurosim-notebooks**: Dedicated notebook server for tutorials (port 8889)
- **quantum-neurosim-api**: Production API server (port 8000)

### Supporting Services

- **qns-db**: PostgreSQL database for experiment tracking (port 5432)
- **qns-redis**: Redis cache for performance optimization (port 6379)
- **qns-prometheus**: Metrics and monitoring (port 9090)

## Environment Variables

### Common Variables

- `PYTHONPATH=/app/src`: Python module path
- `QNS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `QNS_ENV`: Environment mode (development, production)

### Database Configuration

- `POSTGRES_DB=quantum_neurosim`: Database name
- `POSTGRES_USER=qns_user`: Database user
- `POSTGRES_PASSWORD=qns_password`: Database password

## Persistent Volumes

- `qns-data`: Experimental datasets and input files
- `qns-results`: Training results and model outputs
- `qns-models`: Saved quantum neural network models
- `qns-db-data`: PostgreSQL database files
- `qns-redis-data`: Redis persistence files
- `qns-prometheus-data`: Monitoring metrics storage

## Build Targets

### Development

- Includes Jupyter Lab, development tools, and debugging utilities
- Mounts source code for live editing
- Optimized for interactive development

### Production

- Minimal image with only runtime dependencies
- Optimized for performance and security
- Suitable for deployment environments

### Notebook

- Enhanced with visualization libraries and example datasets
- Pre-configured for tutorials and demonstrations
- Includes additional quantum computing packages

## Networking

Services communicate via the `qns-network` bridge network on subnet 172.20.0.0/16.

## Security Notes

- Change default database passwords in production
- Use secrets management for sensitive configuration
- Consider running containers as non-root users
- Regularly update base images for security patches

## Monitoring

Prometheus metrics available at <http://localhost:9090> when monitoring stack is running.

## Troubleshooting

### View Logs

```bash
# View logs for specific service
docker-compose -f docker/docker-compose.yml logs quantum-neurosim-dev

# Follow logs in real-time
docker-compose -f docker/docker-compose.yml logs -f quantum-neurosim-dev
```

### Rebuild Images

```bash
# Rebuild and restart services
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
```

### Clean Up

```bash
# Stop all services and remove containers
docker-compose -f docker/docker-compose.yml down

# Remove volumes (WARNING: This deletes all data)
docker-compose -f docker/docker-compose.yml down -v
```
