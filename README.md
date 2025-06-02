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

## Environment Configuration (.env file)

### Do I need a .env file?

**No, the .env file is optional.** The application will work perfectly fine without it using the default configuration values.

### Default Configuration

If no `.env` file is present, the application uses these defaults:
- `API_BASE_URL`: `https://www.longtermtrends.net`
- `SECRET_KEY`: `None` (not currently required for basic functionality)
- `DEBUG`: `False`

### When to use a .env file

You only need to create a `.env` file if you want to:

1. **Use a different API endpoint** (e.g., for local development or testing)
2. **Enable debug mode** to bypass premium user verification
3. **Set a custom secret key** (for future security features)

### Creating a .env file (Optional)

If you need custom configuration, create a `.env` file in the project root with any of these variables:

```env
# Optional: Change the API base URL (default: https://www.longtermtrends.net)
API_BASE_URL=http://localhost:8000

# Optional: Enable debug mode to grant premium access to all users (default: False)
DEBUG=true

# Optional: Set a secret key for future security features (default: None)
SECRET_KEY=your-secret-key-here
```

### Development vs Production

- **Development**: You might want to set `DEBUG=true` to test premium features
- **Production**: No `.env` file needed, or use it to override the default API endpoint
- **Docker**: Environment variables can be set directly in docker-compose files instead of using `.env`

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

**Tip:**
- Use the existing tabs (such as "US Sectors" or "All-Weather Portfolio") as templates for structure and logic.
- Keep changes minimal and focused on your new tab to maintain code clarity and stability.

## Docker Deployment

This project includes Docker containerization with separate configurations for development, staging, and production environments.

### Prerequisites

- Docker and Docker Compose installed
- Make sure port 8501 is available

- Ensure the deployment script is executable:

  ```bash
  chmod +x deploy.sh
  ```

  (Run this command if you encounter a permission error when trying to execute `./deploy.sh`)

### Quick Start

#### Development Environment

```bash
docker compose -f docker-compose.yml up
```

Access at: <http://localhost:8501>

#### Staging Environment

```bash
./deploy.sh staging

# Or manually
docker compose -f docker-compose.staging.yml up --build -d
```

Access at: <http://localhost:8501>

#### Production Environment

```bash
# Using the deployment script
./deploy.sh prod

# Or manually
docker compose -f docker-compose.prod.yml up --build -d
```

Access at: <http://localhost:8501>


#### Environment-Specific Compose Files

- `docker-compose.yml`: Development with hot reloading and volume mounts (default)
- `docker-compose.staging.yml`: Staging with production-like settings
- `docker-compose.prod.yml`: Production with resource limits and security headers


### Logs

Logs are persisted in the `./logs` directory and mounted into all containers.


#### Check container status

```bash
# Development (default)
docker compose ps

# Staging or Production
docker compose -f docker-compose.[staging|prod].yml ps
```

#### View logs

```bash
# Development (default)
docker compose logs -f

# Staging or Production
docker compose -f docker-compose.[staging|prod].yml logs -f
```

#### Access container shell

env can be `dev`, `staging`, or `prod` depending on the environment you are working with.

```bash
docker exec -it asset-allocation-[env] /bin/bash
```

## Premium User Authentication & Features

This application includes a premium membership system that provides additional analytical capabilities to verified users.

### Premium User Verification Process

The application verifies premium membership through the following process:

1. **Email Parameter**: The user's email is passed as a query parameter in the URL (`?email=user@example.com`)
2. **URL Encoding**: Special characters in the email (like `+`) are properly URL-encoded
3. **API Verification**: The app makes a GET request to verify membership status:
   ```
   GET {API_BASE_URL}/community/verify-user-membership/?email={encoded_email}
   ```
4. **Response Processing**: The API returns a JSON response with an `is_premium_member` boolean field
5. **Session Storage**: The membership status is stored in `st.session_state.is_premium_user`

### Configuration

The premium verification system is controlled by these configuration variables:

- **`API_BASE_URL`**: Base URL for the membership verification API (default: `https://www.longtermtrends.net`)
- **`DEBUG`**: When set to `true`, grants premium access to all users for development purposes

### Authentication Logging

All authentication attempts are logged to `logs/auth.log` with the following information:
- Original and URL-encoded email addresses
- API endpoint calls and responses
- Membership verification results
- Error handling for failed verification attempts

### Premium-Only Content

Premium users gain access to the following exclusive features:

#### Aggregated Performance Metrics
- **Comprehensive Performance Tables**: Detailed performance metrics aggregated by regime and asset
- **Key Metrics Include**:
  - Annualized Return (Aggregated)
  - Annualized Volatility (Aggregated) 
  - Sharpe Ratio (Aggregated)
  - Average Max Drawdown (Period Average)
- **Advanced Visualizations**: Interactive bar charts displaying performance metrics across different economic regimes
- **Regime-Based Analysis**: Color-coded tables and charts that highlight performance during different macroeconomic conditions

#### Features Available on All Analysis Tabs
Premium features are available across all major analysis tabs:
- Asset Classes
- Large vs. Small Cap
- Cyclical vs. Defensive
- US Sectors
- Factor Investing
- All-Weather Portfolio

### Non-Premium User Experience

Users without premium membership can still access:
- Full regime visualization and timeline
- Asset performance charts (normalized to 100)
- Complete trade logs with regime highlighting
- Basic analytical footnotes and explanations

Non-premium users see a membership upgrade prompt where premium content would appear, with a direct link to join the community.

### Development Mode

For development and testing purposes:
- Set `DEBUG=true` in your environment variables
- This grants premium access to all users regardless of membership status
- Useful for testing premium features without API dependencies
