#!/bin/bash
# Stop Quantum Development Environment

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

echo -e "${RED}🛑 Stopping Quantum Development Environment...${NC}"

# Stop all services
echo -e "${BLUE}📦 Stopping containers...${NC}"
docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down

# Option to remove volumes (data will be lost!)
if [[ "$1" == "--remove-volumes" ]] || [[ "$1" == "-v" ]]; then
    echo -e "${RED}⚠️  Removing volumes (this will delete all data)...${NC}"
    read -p "Are you sure you want to delete all data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
        echo -e "${RED}💥 All volumes removed${NC}"
    else
        echo -e "${YELLOW}📦 Volumes preserved${NC}"
    fi
fi

# Option to remove images
if [[ "$1" == "--remove-images" ]] || [[ "$1" == "-i" ]]; then
    echo -e "${BLUE}🗑️  Removing Docker images...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --rmi all
fi

# Clean up dangling resources
if [[ "$2" == "--cleanup" ]] || [[ "$1" == "--cleanup" ]]; then
    echo -e "${BLUE}🧹 Cleaning up Docker resources...${NC}"
    docker system prune -f
    docker volume prune -f
fi

echo -e "${GREEN}✅ Environment stopped successfully!${NC}"
echo ""
echo -e "${BLUE}💡 Available options:${NC}"
echo -e "  --remove-volumes (-v)  Remove all data volumes"
echo -e "  --remove-images (-i)   Remove Docker images"
echo -e "  --cleanup             Clean up dangling resources"
