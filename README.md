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

## Adding a New Tab to the Streamlit App

To add a new analysis tab to the application, follow these steps:

1. **Define the Asset List and Colors**
   - In `config/constants.py`, create a new asset list (e.g., `asset_list_tabX`) with the assets for your new tab.
   - Add color mappings for your new assets to the `asset_colors` dictionary if needed.

2. **Add Data URLs and Fetch Logic**
   - In `data/loader.py`, define URLs for your new assets and update the data loading logic in the `load_data()` function.
   - Fetch, decode, and preprocess each asset’s data as needed (follow the pattern for existing assets).

3. **Include in Data Merge**
   - Append your new asset DataFrames to the main merged DataFrame list in `load_data()` in `data/loader.py` so they are available for analysis and visualization.

4. **Add Tab Name**
   - Add your new tab’s name to the `tabs = st.tabs([...])` list at the desired position in `streamlit_app.py`.

5. **Create Tab Content**
   - Add a new `with tabs[index]:` block for your tab.
   - Set any tab-specific parameters (asset list, minimum assets, checkboxes, etc.).
   - Call `render_asset_analysis_tab` (or similar) with the correct arguments for your new tab, following the structure of existing tabs.

6. **Test the App**
   - Run the app and verify that your new tab appears and functions correctly.

**Note:**
- The data loading logic and all asset data URLs have been moved from `streamlit_app.py` to `data/loader.py`. If you need to add or edit asset URLs or data fetching logic, do so in `data/loader.py`.

**Tip:**
- Use the existing tabs (such as "US Sectors" or "All-Weather Portfolio") as templates for structure and logic.
- Keep changes minimal and focused on your new tab to maintain code clarity and stability.