# longtermtrends

## Setup Instructions

Before running the application, create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application (in the virtual environment)

```bash
python3 -m streamlit run streamlit_app.py
```

## Refactored App Structure (April 2025)

### Key Changes
- **Modular Tab System:**
  - All major analysis tabs are now implemented as separate modules in the `tabs/` directory. Each tab has a dedicated `render` function, making it easier to maintain and extend the app.
- **Consistent User Controls:**
  - Tabs that support asset/trade filtering (e.g., Factor Investing, Large vs. Small Cap, US Sectors) now feature two checkboxes:
    - **Include trades before cutoff:** Allows users to include trades before a dynamically calculated cutoff date. The label displays the cutoff date and minimum asset requirement.
    - **Include late-start assets:** Allows inclusion of assets that started after the cutoff date.
- **Restored and Standardized Footnotes:**
  - Calculation and aggregation notes are now shown directly below the Aggregated Performance Metrics table on all relevant tabs for transparency.
- **Regime Visualization Improvements:**
  - The regime timeline, legend, and scatter plot sections have been modularized and visually improved.
- **Improved Codebase Readability:**
  - All tab logic is separated from the main app file, and shared logic is centralized.

### Directory Structure
```
streamlit_app.py
core/
  asset_analysis.py
  charts.py
  constants.py
  fetch.py
  loader.py
  performance.py
  processing.py
tabs/
  all_weather.py
  asset_classes.py
  cyclical_vs_defensive.py
  factor_investing.py
  large_vs_small.py
  regime.py
  us_sectors.py
Ke/
  Visualization-1973.ipynb
  input_preprocessing.ipynb
  week 9_Ke_2nd version.docx
requirements.txt
README.md
```

### File/Directory Purpose

- **streamlit_app.py**: Main entry point for the Streamlit app; orchestrates data loading, tab rendering, and global configuration.
- **core/**: Contains all core logic and utilities:
  - **asset_analysis.py**: Shared logic for rendering asset analysis tabs and computing dynamic cutoffs.
  - **charts.py**: Functions for generating all charts and visualizations.
  - **constants.py**: Centralized constants for assets, regimes, colors, and labels.
  - **fetch.py**: Fetches and decodes remote data for use in the app.
  - **loader.py**: Loads and prepares all datasets needed by the app.
  - **performance.py**: Functions for calculating performance metrics and trade logs.
  - **processing.py**: Data processing utilities (merging, moving averages, regime assignment, etc.).
- **tabs/**: Contains the logic for each user-facing analysis tab:
  - **all_weather.py**: All-Weather Portfolio tab.
  - **asset_classes.py**: Major Asset Classes tab.
  - **cyclical_vs_defensive.py**: Cyclical vs. Defensive Stocks tab.
  - **factor_investing.py**: Factor Investing tab.
  - **large_vs_small.py**: Large vs. Small Cap tab.
  - **regime.py**: Regime Visualization tab.
  - **us_sectors.py**: US Sectors tab.
- **requirements.txt**: Lists all Python dependencies required to run the app.
- **README.md**: Project overview, setup instructions, and documentation.

## Adding a New Tab to the Streamlit App

To add a new analysis tab to the application, follow these steps:

1. **Create a new file in `tabs/` (e.g., `tabs/my_new_tab.py`).**
2. **Implement a `render(tab, asset_ts_data, sp_inflation_data, session_state)` function.**
3. **Import and call your new tab’s render function from `streamlit_app.py`.**

When using charts, import them from `core.charts` like so:
```python
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
```

### Recent Improvements
- Modularized all tabs for clarity and scalability.
- Standardized user controls and footnotes.
- Improved regime visualization and legend placement.
- Enhanced maintainability and extensibility of the codebase.

---

## Docker Deployment

This project includes Docker containerization with separate configurations for development, staging, and production environments.

### Prerequisites

- Docker and Docker Compose installed
- Make sure ports 8501, 8502, and 8503 are available

### Quick Start

#### Development Environment

```bash
# Quick start with default compose file
docker compose up --build

# Or using the deployment script
./deploy.sh dev

# Or manually specifying the file
docker-compose -f docker-compose.yml up --build
```

Access at: <http://localhost:8501>

#### Staging Environment

```bash
# Using the deployment script
./deploy.sh staging

# Or manually
docker-compose -f docker-compose.staging.yml up --build -d
```

Access at: <http://localhost:8502>

#### Production Environment

```bash
# Using the deployment script
./deploy.sh prod

# Or manually
docker-compose -f docker-compose.prod.yml up --build -d
```

Access at: <http://localhost:8503>

### Environment Configuration

The application uses a single `.env` file for all environments with environment-specific overrides in the Docker Compose files.

#### Environment Variables in `.env`

- `DEBUG`: Enable/disable debug mode (overridden per environment)
- `API_BASE_URL`: Backend API URL (overridden per environment)
- `STREAMLIT_ENV`: Environment identifier (set by Docker Compose)
- `SECRET_KEY`: Application secret key
- `STREAMLIT_SERVER_PORT`: Streamlit server port
- `STREAMLIT_SERVER_ADDRESS`: Streamlit server address

#### Environment-Specific Overrides

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

### Docker Files

#### Core Files

- `Dockerfile`: Base container configuration
- `.dockerignore`: Files to exclude from Docker context

#### Environment-Specific Compose Files

- `docker-compose.yml`: Development with hot reloading and volume mounts (default)
- `docker-compose.staging.yml`: Staging with production-like settings
- `docker-compose.prod.yml`: Production with resource limits and security headers

### Features by Environment

#### Development

- Hot reloading enabled
- Source code mounted as volume
- Debug mode enabled
- Port 8501

#### Staging

- Production-like configuration
- No source code mounting
- SSL/TLS ready with Traefik labels
- Port 8502

#### Production

- Resource limits (2GB memory, 2 CPU)
- Security headers enabled
- CORS disabled
- XSRF protection enabled
- SSL/TLS ready with Traefik labels
- Port 8503

### Deployment Script

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

### Health Monitoring

All containers include health checks that verify the Streamlit application is responding on the `/_stcore/health` endpoint.

### Logs

Logs are persisted in the `./logs` directory and mounted into all containers.

### Network

All containers use a custom Docker network (`asset-allocation-network`) for isolation and internal communication.

### Production Considerations

For production deployment, consider:

1. Using a reverse proxy (Nginx, Traefik) for SSL termination
2. Setting up proper monitoring and logging
3. Configuring backup strategies for persistent data
4. Implementing proper secrets management
5. Setting up CI/CD pipelines for automated deployments

### Troubleshooting

#### Check container status

```bash
# Development (default)
docker-compose ps

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml ps
```

#### View logs

```bash
# Development (default)
docker-compose logs -f

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml logs -f
```

#### Access container shell

```bash
docker exec -it asset-allocation-[env] /bin/bash
```

#### Health check manually

```bash
curl http://localhost:[port]/_stcore/health
```

**Tip:**
- Use the existing tabs (such as "US Sectors" or "All-Weather Portfolio") as templates for structure and logic.
- Keep changes minimal and focused on your new tab to maintain code clarity and stability.

---

## Docker Deployment

This project includes Docker containerization with separate configurations for development, staging, and production environments.

### Prerequisites
- Docker and Docker Compose installed
- Make sure ports 8501, 8502, and 8503 are available

### Quick Start

#### Development Environment
```bash
# Quick start with default compose file
docker compose up --build

# Or using the deployment script
./deploy.sh dev

# Or manually specifying the file
docker-compose -f docker-compose.yml up --build
```
Access at: http://localhost:8501

#### Staging Environment
```bash
# Using the deployment script
./deploy.sh staging

# Or manually
docker-compose -f docker-compose.staging.yml up --build -d
```
Access at: http://localhost:8502

#### Production Environment
```bash
# Using the deployment script
./deploy.sh prod

# Or manually
docker-compose -f docker-compose.prod.yml up --build -d
```
Access at: http://localhost:8503

### Environment Configuration

The application uses a single `.env` file for all environments with environment-specific overrides in the Docker Compose files.

#### Environment Variables in `.env`
- `DEBUG`: Enable/disable debug mode (overridden per environment)
- `API_BASE_URL`: Backend API URL (overridden per environment)
- `STREAMLIT_ENV`: Environment identifier (set by Docker Compose)
- `SECRET_KEY`: Application secret key
- `STREAMLIT_SERVER_PORT`: Streamlit server port
- `STREAMLIT_SERVER_ADDRESS`: Streamlit server address

#### Environment-Specific Overrides
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

### Docker Files

#### Core Files
- `Dockerfile`: Base container configuration
- `.dockerignore`: Files to exclude from Docker context

#### Environment-Specific Compose Files
- `docker-compose.yml`: Development with hot reloading and volume mounts (default)
- `docker-compose.staging.yml`: Staging with production-like settings
- `docker-compose.prod.yml`: Production with resource limits and security headers

### Features by Environment

#### Development
- Hot reloading enabled
- Source code mounted as volume
- Debug mode enabled
- Port 8501

#### Staging
- Production-like configuration
- No source code mounting
- SSL/TLS ready with Traefik labels
- Port 8502

#### Production
- Resource limits (2GB memory, 2 CPU)
- Security headers enabled
- CORS disabled
- XSRF protection enabled
- SSL/TLS ready with Traefik labels
- Port 8503

### Deployment Script

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

### Health Monitoring

All containers include health checks that verify the Streamlit application is responding on the `/_stcore/health` endpoint.

### Logs

Logs are persisted in the `./logs` directory and mounted into all containers.

### Network

All containers use a custom Docker network (`asset-allocation-network`) for isolation and internal communication.

### Production Considerations

For production deployment, consider:
1. Using a reverse proxy (Nginx, Traefik) for SSL termination
2. Setting up proper monitoring and logging
3. Configuring backup strategies for persistent data
4. Implementing proper secrets management
5. Setting up CI/CD pipelines for automated deployments

### Troubleshooting

#### Check container status
```bash
# Development (default)
docker-compose ps

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml ps
```

#### View logs
```bash
# Development (default)
docker-compose logs -f

# Staging or Production
docker-compose -f docker-compose.[staging|prod].yml logs -f
```

#### Access container shell
```bash
docker exec -it asset-allocation-[env] /bin/bash
```

#### Health check manually
```bash
curl http://localhost:[port]/_stcore/health
```