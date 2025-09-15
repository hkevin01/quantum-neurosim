#!/bin/bash
# Quantum Development Environment Management Script
# Start quantum development environment with Docker

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker/docker-compose.advanced.yml"
PROJECT_NAME="quantum-neurosim"

echo -e "${BLUE}🚀 Starting Quantum Development Environment...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA runtime is available for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected - CPU-only mode${NC}"
    GPU_AVAILABLE=false
fi

# Build images if they don't exist or if --rebuild flag is provided
if [[ "$1" == "--rebuild" ]] || [[ "$1" == "-r" ]]; then
    echo -e "${BLUE}🔨 Building Docker images...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build --no-cache
else
    echo -e "${BLUE}🔨 Building Docker images (cached)...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data results notebooks/tutorials notebooks/experiments
mkdir -p monitoring/grafana/dashboards monitoring/grafana/provisioning
mkdir -p config sql/init

# Start core services (CPU development environment)
echo -e "${BLUE}🟢 Starting core quantum development services...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d quantum-dev quantum-db quantum-redis

# Start GPU service if available and requested
if [[ "$GPU_AVAILABLE" == true ]] && [[ "$2" == "--gpu" ]]; then
    echo -e "${BLUE}🟢 Starting GPU-accelerated quantum environment...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME --profile gpu up -d quantum-gpu
fi

# Start monitoring if requested
if [[ "$2" == "--monitoring" ]] || [[ "$3" == "--monitoring" ]]; then
    echo -e "${BLUE}📊 Starting monitoring services...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME --profile monitoring up -d
fi

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to initialize...${NC}"
sleep 15

# Check service health
echo -e "${BLUE}🔍 Checking service health...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps

# Display access information
echo -e "${GREEN}✅ Quantum Development Environment Ready!${NC}"
echo ""
echo -e "${BLUE}📊 Access Points:${NC}"
echo -e "  🔬 Jupyter Lab (CPU):     ${GREEN}http://localhost:8888${NC}"

if [[ "$GPU_AVAILABLE" == true ]] && [[ "$2" == "--gpu" ]]; then
    echo -e "  🚀 Jupyter Lab (GPU):     ${GREEN}http://localhost:8889${NC}"
fi

echo -e "  🗄️  PostgreSQL Database:  ${GREEN}localhost:5432${NC}"
echo -e "  💾 Redis Cache:           ${GREEN}localhost:6379${NC}"

if [[ "$2" == "--monitoring" ]] || [[ "$3" == "--monitoring" ]]; then
    echo -e "  📈 Prometheus:            ${GREEN}http://localhost:9090${NC}"
    echo -e "  📊 Grafana:               ${GREEN}http://localhost:3000${NC} (admin/quantum123)"
fi

echo ""
echo -e "${BLUE}🎯 Next Steps:${NC}"
echo -e "  1. Open Jupyter Lab in your browser"
echo -e "  2. Navigate to notebooks/quantum_docker_tutorial.ipynb for the Docker guide"
echo -e "  3. Run examples/01_basic_classification.py to test the framework"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "  • Use 'docker-compose -f $COMPOSE_FILE logs -f' to view logs"
echo -e "  • Use './scripts/stop.sh' to stop all services"
echo -e "  • Use './scripts/backup.sh' to backup experiment data"

# Create .env file with placeholder values if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}📝 Creating .env file with placeholders...${NC}"
    cat > .env << EOF
# Quantum Cloud Provider Configuration
# IBM Quantum
QISKIT_IBM_TOKEN=your_ibm_quantum_token_here
QISKIT_IBM_CHANNEL=ibm_quantum

# AWS Braket
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Database Configuration
POSTGRES_DB=quantum_experiments
POSTGRES_USER=quantum_user
POSTGRES_PASSWORD=quantum_pass

# Application Settings
QUANTUM_LOG_LEVEL=INFO
QN_ENV=development
EOF
    echo -e "${GREEN}✅ Created .env file - please update with your credentials${NC}"
fi

echo -e "${GREEN}🎉 Setup complete! Happy quantum computing!${NC}"
