#!/bin/bash
# Backup Quantum Experiment Data

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
BACKUP_BASE_DIR="backups"

# Create backup directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE_DIR/quantum_backup_$TIMESTAMP"

echo -e "${BLUE}💾 Starting Quantum Experiment Data Backup...${NC}"
echo -e "${BLUE}📁 Backup destination: $BACKUP_DIR${NC}"

# Create backup directory
mkdir -p "$BACKUP_DIR"/{database,volumes,notebooks,results,config}

# Check if containers are running
if ! docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  No running containers found. Starting minimal services for backup...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d quantum-db quantum-redis
    sleep 5
    STARTED_FOR_BACKUP=true
fi

# Backup PostgreSQL database
echo -e "${BLUE}🗄️  Backing up PostgreSQL database...${NC}"
if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T quantum-db pg_isready -U quantum_user -d quantum_experiments > /dev/null 2>&1; then
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T quantum-db \
        pg_dump -U quantum_user quantum_experiments > "$BACKUP_DIR/database/quantum_experiments.sql"
    echo -e "${GREEN}✅ Database backup completed${NC}"
else
    echo -e "${RED}❌ Database not available for backup${NC}"
fi

# Backup Redis data
echo -e "${BLUE}💾 Backing up Redis data...${NC}"
if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T quantum-redis redis-cli ping > /dev/null 2>&1; then
    docker run --rm \
        -v quantum-neurosim_quantum-redis-data:/source:ro \
        -v "$(pwd)/$BACKUP_DIR/volumes":/backup \
        alpine tar czf /backup/redis-data.tar.gz -C /source .
    echo -e "${GREEN}✅ Redis backup completed${NC}"
else
    echo -e "${RED}❌ Redis not available for backup${NC}"
fi

# Backup Docker volumes
echo -e "${BLUE}📦 Backing up Docker volumes...${NC}"
VOLUMES=(
    "quantum-cache:cache-data.tar.gz"
    "quantum-gpu-cache:gpu-cache-data.tar.gz"
    "quantum-metrics:metrics-data.tar.gz"
    "grafana-data:grafana-data.tar.gz"
)

for volume_info in "${VOLUMES[@]}"; do
    volume_name=$(echo $volume_info | cut -d: -f1)
    backup_file=$(echo $volume_info | cut -d: -f2)

    if docker volume ls | grep -q "${PROJECT_NAME}_${volume_name}"; then
        echo -e "${BLUE}  📁 Backing up volume: $volume_name${NC}"
        docker run --rm \
            -v "${PROJECT_NAME}_${volume_name}":/source:ro \
            -v "$(pwd)/$BACKUP_DIR/volumes":/backup \
            alpine tar czf "/backup/$backup_file" -C /source . 2>/dev/null || echo -e "${YELLOW}    ⚠️ Volume $volume_name is empty or unavailable${NC}"
    fi
done

# Backup notebooks and results directories
echo -e "${BLUE}📓 Backing up notebooks...${NC}"
if [[ -d "notebooks" ]]; then
    cp -r notebooks "$BACKUP_DIR/"
    echo -e "${GREEN}✅ Notebooks backup completed${NC}"
fi

echo -e "${BLUE}📊 Backing up results...${NC}"
if [[ -d "results" ]]; then
    cp -r results "$BACKUP_DIR/"
    echo -e "${GREEN}✅ Results backup completed${NC}"
fi

# Backup configuration files
echo -e "${BLUE}⚙️  Backing up configuration...${NC}"
CONFIG_FILES=(
    "docker/docker-compose.advanced.yml"
    "docker/Dockerfile.cpu-quantum"
    "docker/Dockerfile.gpu-quantum"
    "docker/requirements-quantum.txt"
    ".env"
    "config/"
    "monitoring/"
)

for config in "${CONFIG_FILES[@]}"; do
    if [[ -e "$config" ]]; then
        cp -r "$config" "$BACKUP_DIR/config/" 2>/dev/null || true
    fi
done

# Create backup manifest
echo -e "${BLUE}📋 Creating backup manifest...${NC}"
cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" << EOF
Quantum NeuroSim Backup
=======================
Timestamp: $TIMESTAMP
Backup Directory: $BACKUP_DIR
Host: $(hostname)
Docker Version: $(docker --version)
User: $(whoami)

Contents:
---------
$(find "$BACKUP_DIR" -type f | sort)

Database Info:
--------------
Database: quantum_experiments
User: quantum_user

Volume Info:
------------
$(docker volume ls | grep $PROJECT_NAME || echo "No project volumes found")

Size Information:
-----------------
$(du -sh "$BACKUP_DIR"/* 2>/dev/null | sort -hr)
EOF

# Stop services if we started them for backup
if [[ "$STARTED_FOR_BACKUP" == true ]]; then
    echo -e "${BLUE}🛑 Stopping services started for backup...${NC}"
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
fi

# Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)

echo -e "${GREEN}✅ Backup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Backup Summary:${NC}"
echo -e "  📁 Location: $BACKUP_DIR"
echo -e "  📏 Size: $BACKUP_SIZE"
echo -e "  ⏰ Timestamp: $TIMESTAMP"
echo ""
echo -e "${BLUE}🔄 To restore from this backup:${NC}"
echo -e "  1. Stop all services: ./scripts/stop.sh"
echo -e "  2. Restore database: cat $BACKUP_DIR/database/quantum_experiments.sql | docker exec -i quantum-db psql -U quantum_user -d quantum_experiments"
echo -e "  3. Restore volumes using Docker volume commands"
echo -e "  4. Copy notebooks and results back to project directory"
echo ""
echo -e "${YELLOW}💡 Tip: Consider creating automated backups with cron jobs${NC}"
