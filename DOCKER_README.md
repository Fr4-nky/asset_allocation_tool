# Docker Setup for Asset Allocation Tool

This project includes Docker containerization with separate configurations for development, staging, and production environments.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Make sure ports 8501, 8502, and 8503 are available

### Development Environment
```bash
# Using the deployment script
./deploy.sh dev

# Or manually
docker-compose -f docker-compose.dev.yml up --build
```
Access at: http://localhost:8501

### Staging Environment
```bash
# Using the deployment script
./deploy.sh staging

# Or manually
docker-compose -f docker-compose.staging.yml up --build -d
```
Access at: http://localhost:8502

### Production Environment
```bash
# Using the deployment script
./deploy.sh prod

# Or manually
docker-compose -f docker-compose.prod.yml up --build -d
```
Access at: http://localhost:8503

## Environment Configuration

Each environment has its own configuration file:
- `.env.dev` - Development settings
- `.env.staging` - Staging settings  
- `.env.prod` - Production settings

### Key Environment Variables
- `DEBUG`: Enable/disable debug mode
- `API_BASE_URL`: Backend API URL
- `STREAMLIT_ENV`: Environment identifier
- `SECRET_KEY`: Application secret key

## Docker Files

### Core Files
- `Dockerfile`: Base container configuration
- `.dockerignore`: Files to exclude from Docker context

### Environment-Specific Compose Files
- `docker-compose.dev.yml`: Development with hot reloading and volume mounts
- `docker-compose.staging.yml`: Staging with production-like settings
- `docker-compose.prod.yml`: Production with resource limits and security headers

## Features by Environment

### Development
- Hot reloading enabled
- Source code mounted as volume
- Debug mode enabled
- Port 8501

### Staging
- Production-like configuration
- No source code mounting
- SSL/TLS ready with Traefik labels
- Port 8502

### Production
- Resource limits (2GB memory, 2 CPU)
- Security headers enabled
- CORS disabled
- XSRF protection enabled
- SSL/TLS ready with Traefik labels
- Port 8503

## Deployment Script

The `deploy.sh` script provides a convenient way to deploy any environment:

```bash
./deploy.sh [dev|staging|prod]
```

Features:
- Environment validation
- Docker health checks
- Container status reporting
- Automatic log display
- Health check verification

## Health Monitoring

All containers include health checks that verify the Streamlit application is responding on the `/_stcore/health` endpoint.

## Logs

Logs are persisted in the `./logs` directory and mounted into all containers.

## Network

All containers use a custom Docker network (`asset-allocation-network`) for isolation and internal communication.

## Production Considerations

For production deployment, consider:
1. Using a reverse proxy (Nginx, Traefik) for SSL termination
2. Setting up proper monitoring and logging
3. Configuring backup strategies for persistent data
4. Implementing proper secrets management
5. Setting up CI/CD pipelines for automated deployments

## Troubleshooting

### Check container status
```bash
docker-compose -f docker-compose.[env].yml ps
```

### View logs
```bash
docker-compose -f docker-compose.[env].yml logs -f
```

### Access container shell
```bash
docker exec -it asset-allocation-[env] /bin/bash
```

### Health check manually
```bash
curl http://localhost:[port]/_stcore/health
```
