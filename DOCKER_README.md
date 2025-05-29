# Docker Setup for Asset Allocation Tool

This project includes Docker containerization with separate configurations for development, staging, and production environments.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Make sure ports 8501, 8502, and 8503 are available

### Development Environment
```bash
# Quick start with default compose file
docker compose up --build

# Or using the deployment script
./deploy.sh dev

# Or manually specifying the file
docker-compose -f docker-compose.yml up --build
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

The application uses a single `.env` file for all environments with environment-specific overrides in the Docker Compose files.

### Environment Variables in `.env`
- `DEBUG`: Enable/disable debug mode (overridden per environment)
- `API_BASE_URL`: Backend API URL (overridden per environment)
- `STREAMLIT_ENV`: Environment identifier (set by Docker Compose)
- `SECRET_KEY`: Application secret key
- `STREAMLIT_SERVER_PORT`: Streamlit server port
- `STREAMLIT_SERVER_ADDRESS`: Streamlit server address

### Environment-Specific Overrides
Each Docker Compose file overrides specific variables:

**Development (`docker-compose.yml`)**:
- `DEBUG=true`
- `API_BASE_URL=http://localhost:8000`
- `STREAMLIT_ENV=development`

**Staging (`docker-compose.staging.yml`)**:
- `DEBUG=false`
- `API_BASE_URL=https://staging.longtermtrends.com`
- `STREAMLIT_ENV=staging`

**Production (`docker-compose.prod.yml`)**:
- `DEBUG=false`
- `API_BASE_URL=https://www.longtermtrends.com`
- `STREAMLIT_ENV=production`

## Docker Files

### Core Files
- `Dockerfile`: Base container configuration
- `.dockerignore`: Files to exclude from Docker context

### Environment-Specific Compose Files
- `docker-compose.yml`: Development with hot reloading and volume mounts (default)
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
# Development (default)
docker-compose ps

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml ps
```

### View logs
```bash
# Development (default)
docker-compose logs -f

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml logs -f
```

### Access container shell
```bash
docker exec -it asset-allocation-[env] /bin/bash
```

### Health check manually
```bash
curl http://localhost:[port]/_stcore/health
```
