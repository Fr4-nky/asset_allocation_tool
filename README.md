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
python -m streamlit run streamlit_app.py
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
  - All tab logic is separated from the main app file, and shared logic is centralized in `ui/asset_analysis.py`.

### Directory Structure
```
streamlit_app.py
metrics/
  performance.py
tabs/
  all_weather.py
  asset_classes.py
  cyclical_vs_defensive.py
  factor_investing.py
  large_vs_small.py
  regime.py
  us_sectors.py
ui/
  asset_analysis.py
...
```

## Adding a New Tab to the Streamlit App

To add a new analysis tab to the application, follow these steps:

1. **Create a new file in `tabs/` (e.g., `tabs/my_new_tab.py`).**
2. **Implement a `render(tab, asset_ts_data, sp_inflation_data, session_state)` function.**
3. **Import and call your new tab’s render function from `streamlit_app.py`.**

### Recent Improvements
- Modularized all tabs for clarity and scalability.
- Standardized user controls and footnotes.
- Improved regime visualization and legend placement.
- Enhanced maintainability and extensibility of the codebase.

**Note:**
- The data loading logic and all asset data URLs have been moved from `streamlit_app.py` to `data/loader.py`. If you need to add or edit asset URLs or data fetching logic, do so in `data/loader.py`.

**Tip:**
- Use the existing tabs (such as "US Sectors" or "All-Weather Portfolio") as templates for structure and logic.
- Keep changes minimal and focused on your new tab to maintain code clarity and stability.