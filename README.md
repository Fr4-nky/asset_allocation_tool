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

**Tip:**
- Use the existing tabs (such as "US Sectors" or "All-Weather Portfolio") as templates for structure and logic.
- Keep changes minimal and focused on your new tab to maintain code clarity and stability.