#!/bin/bash

# Docker deployment script for Asset Allocation Tool
# Usage: ./deploy.sh [dev|staging|prod]

set -e

ENVIRONMENT=${1:-dev}
PROJECT_NAME="asset-allocation-tool"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Asset Allocation Tool - ${ENVIRONMENT} environment${NC}"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo -e "${RED}❌ Error: Environment must be 'dev', 'staging', or 'prod'${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Using environment variables from .env file${NC}"

# Stop and remove existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.${ENVIRONMENT}.yml down --remove-orphans

# Build and start containers
echo -e "${YELLOW}🔨 Building and starting containers...${NC}"
docker-compose -f docker-compose.${ENVIRONMENT}.yml up --build -d

# Show container status
echo -e "${GREEN}📊 Container status:${NC}"
docker-compose -f docker-compose.${ENVIRONMENT}.yml ps

# Show logs
echo -e "${GREEN}📋 Recent logs:${NC}"
docker-compose -f docker-compose.${ENVIRONMENT}.yml logs --tail=10

# Health check
echo -e "${YELLOW}🏥 Performing health check...${NC}"
sleep 10

CONTAINER_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
if [ "$ENVIRONMENT" = "dev" ]; then
    PORT=8501
elif [ "$ENVIRONMENT" = "staging" ]; then
    PORT=8502
else
    PORT=8503
fi

# Check if the application is responding
if curl -f http://localhost:${PORT}/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Application is healthy and running on port ${PORT}${NC}"
    echo -e "${GREEN}🌐 Access the application at: http://localhost:${PORT}${NC}"
else
    echo -e "${RED}❌ Health check failed. Check logs:${NC}"
    docker-compose -f docker-compose.${ENVIRONMENT}.yml logs --tail=20
fi

echo -e "${GREEN}🎉 Deployment complete!${NC}"
