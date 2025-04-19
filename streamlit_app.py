# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io  # For exporting plot as image
import requests # Added for API calls
import json     # Added for API calls
import base64   # Added for API calls
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import time

# Set page configuration
st.set_page_config(
    page_title="Macroeconomic Regimes and Asset Performance",
    layout="wide",
    page_icon="https://www.longtermtrends.net/static/my_app/images/favicon.ico"
)

# --- Add meta tags for SEO and social sharing ---
meta_title = "Macroeconomic Regimes and Asset Performance Analysis | Longtermtrends"
meta_description = ("This app visualizes macroeconomic regimes based on S&P 500 and Inflation Rate data, and analyzes asset performance across these regimes.")
# Inject meta tags into the Streamlit app (HTML head)
st.markdown(f"""
    <meta name='title' content='{meta_title}'>
    <meta name='description' content='{meta_description}'>
    <meta property='og:title' content='{meta_title}'>
    <meta property='og:description' content='{meta_description}'>
    <meta property='og:type' content='website'>
""", unsafe_allow_html=True)

# Title and Description
st.title("Macroeconomic Regimes and Asset Performance Analysis")
st.write("""
This app visualizes macroeconomic regimes based on S&P 500 and Inflation Rate data, and analyzes asset performance across these regimes.
""")

# --- Helper Functions for API Data Fetching ---

def decode_base64_data(encoded_data):
    """Decodes a list of [base64_date, base64_value] pairs."""
    decoded_list = []
    for date_b64, value_b64 in encoded_data:
        try:
            date_str = base64.b64decode(date_b64).decode('utf-8')
            value_str = base64.b64decode(value_b64).decode('utf-8')
            # Convert value to float, handle potential errors (e.g., non-numeric values)
            value_float = float(value_str)
            decoded_list.append([date_str, value_float])
        except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
            # Use print for terminal output instead of st.warning
            print(f"WARNING: Skipping record due to decoding/conversion error: {e} - Date: {date_b64}, Value: {value_b64}")
            # Optionally append with None or np.nan if you want to keep the row
            # date_str = base64.b64decode(date_b64).decode('utf-8', errors='ignore')
            # decoded_list.append([date_str, None])
    return decoded_list

def fetch_and_decode(url, column_name):
    """Fetches data from a URL, decodes it, and returns a Pandas DataFrame."""
    # Use print for terminal output instead of st.info
    print(f"INFO: Fetching data from {url}...")
    try:
        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        encoded_data = response.json()
        decoded_data = decode_base64_data(encoded_data)

        if not decoded_data: # Handle case where decoding resulted in an empty list
             # Use print for terminal output instead of st.warning
             print(f"WARNING: No valid data decoded from {url}")
             return None

        df = pd.DataFrame(decoded_data, columns=['Date', column_name])
        df['Date'] = pd.to_datetime(df['Date']) # Convert Date column to datetime objects
        df = df.set_index('Date') # Set Date as index for easy merging
        # Use print for terminal output instead of st.success
        print(f"SUCCESS: Successfully fetched and processed data for {column_name}.")
        return df
    except requests.exceptions.Timeout:
        # Use print for terminal output instead of st.error
        print(f"ERROR: Request timed out while fetching data from {url}")
        return None
    except requests.exceptions.RequestException as e:
        # Use print for terminal output instead of st.error
        print(f"ERROR: Error fetching data from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        # Use print for terminal output instead of st.error
        print(f"ERROR: Error decoding JSON from {url}. Response text: {response.text[:500]}... Error: {e}") # Log part of the response
        return None
    except Exception as e:
        # Use print for terminal output instead of st.error
        print(f"ERROR: An unexpected error occurred while processing {url}: {e}")
        return None

# --- End Helper Functions ---

# Function to merge asset data with regime assignments
def merge_asset_with_regimes(asset_ts_df, sp_inflation_filtered):
    """
    Merge asset time series data with regime assignments based on DateTime.
    Assumes both DataFrames have a 'DateTime' column.
    Adds 'Regime' to asset_ts_df by merging on 'DateTime'.
    Also adds a 'Regime_Change' column to mark regime transitions.
    """
    # Merge asset data with regime info on DateTime
    merged = pd.merge(asset_ts_df, sp_inflation_filtered[['DateTime', 'Regime']], on='DateTime', how='left')
    # Fill missing regimes as 'Unknown'
    merged['Regime'] = merged['Regime'].fillna('Unknown')
    # Add a column to mark regime changes (for background shading)
    merged = merged.sort_values('DateTime').reset_index(drop=True)
    merged['Regime_Change'] = (merged['Regime'] != merged['Regime'].shift(1)).cumsum()
    return merged

# Load Data Function
@st.cache_data
def load_data():
    # --- Fetch S&P 500, Inflation from API ---
    sp500_url = "https://www.longtermtrends.net/data-sp500-since-1871/"
    inflation_url = "https://www.longtermtrends.net/data-inflation-forecast/"
    bonds_url = "https://www.longtermtrends.net/data-total-return-bond-index/"
    gold_url = "https://www.longtermtrends.net/data-gold-since-1792/"

    # Fetch each dataset (returns df with 'Date' as index or None)
    df_sp500 = fetch_and_decode(sp500_url, 'S&P 500')
    df_inflation = fetch_and_decode(inflation_url, 'Inflation Rate')
    df_bonds = fetch_and_decode(bonds_url, 'Bonds')
    df_gold = fetch_and_decode(gold_url, 'Gold')

    # --- Data Preprocessing (Resampling, Merging, Filtering) ---
    print("DEBUG: Applying resampling and merging logic...")
    today = pd.Timestamp.today().normalize()
    print(f"DEBUG: Current date (normalized): {today}")

    # 1. Resample All Series to Business Month End ('BME') and Correct Future Dates
    def resample_and_correct_date(df, name):
        if df is not None:
            print(f"DEBUG: Resampling {name} (initial shape: {df.shape})")
            print(f"DEBUG: {name} last date before resample: {df.index.max()}")
            df_resampled = df.resample('BME').last()
            print(f"DEBUG: Resampled {name} shape: {df_resampled.shape}")
            print(f"DEBUG: {name} last date after resample: {df_resampled.index.max()}")
            if not df_resampled.empty and df_resampled.index.max() > today:
                print(f"WARN: {name} last date {df_resampled.index.max()} is after today {today}. Correcting...")
                # Use index.where to replace future dates, ensuring index name is preserved
                original_index_name = df_resampled.index.name
                df_resampled.index = df_resampled.index.where(df_resampled.index <= today, today)
                df_resampled.index.name = original_index_name # Restore index name if lost
                print(f"DEBUG: {name} last date after correction: {df_resampled.index.max()}")
            return df_resampled
        return None

    df_sp500_resampled = resample_and_correct_date(df_sp500, 'S&P 500')
    df_inflation_resampled = resample_and_correct_date(df_inflation, 'Inflation Rate')
    df_bonds_resampled = resample_and_correct_date(df_bonds, 'Bonds')
    df_gold_resampled = resample_and_correct_date(df_gold, 'Gold')

    # 2. Inner Merge S&P 500 and Inflation Rate (for Tab 1)
    sp_inflation_df = pd.DataFrame() # Initialize empty df
    if df_sp500_resampled is not None and df_inflation_resampled is not None:
        print("DEBUG: Performing INNER merge on resampled S&P 500 and Inflation Rate...")
        print(f"DEBUG: S&P 500 index min: {df_sp500_resampled.index.min()}, max: {df_sp500_resampled.index.max()}")
        print(f"DEBUG: Inflation Rate index min: {df_inflation_resampled.index.min()}, max: {df_inflation_resampled.index.max()}")
        # Ensure index names match before merge if they exist
        if df_sp500_resampled.index.name != df_inflation_resampled.index.name:
             print(f"WARN: Index names differ ('{df_sp500_resampled.index.name}' vs '{df_inflation_resampled.index.name}'). Aligning index name for merge.")
             # Decide on a common name, e.g., 'Date' or use the first df's name
             common_index_name = df_sp500_resampled.index.name if df_sp500_resampled.index.name else 'Date'
             df_sp500_resampled.index.name = common_index_name
             df_inflation_resampled.index.name = common_index_name
        sp_inflation_df = pd.merge(df_sp500_resampled, df_inflation_resampled, left_index=True, right_index=True, how='inner')
        print(f"DEBUG: Inner merge result shape: {sp_inflation_df.shape}")
        print(f"DEBUG: Inner merge index min: {sp_inflation_df.index.min()}, max: {sp_inflation_df.index.max()}")
        print(f"DEBUG: Number of rows with non-null S&P 500: {(~sp_inflation_df['S&P 500'].isnull()).sum()}")
        print(f"DEBUG: Number of rows with non-null Inflation Rate: {(~sp_inflation_df['Inflation Rate'].isnull()).sum()}")
        print(f"DEBUG: Last row with non-null values: \n{sp_inflation_df.dropna().tail(1)}")
    elif df_sp500_resampled is not None:
        print("WARN: Inflation data missing, using only S&P 500 data.")
        sp_inflation_df = df_sp500_resampled.copy()
        sp_inflation_df['Inflation Rate'] = np.nan # Add missing column
    elif df_inflation_resampled is not None:
        print("WARN: S&P 500 data missing, using only Inflation data.")
        sp_inflation_df = df_inflation_resampled.copy()
        sp_inflation_df['S&P 500'] = np.nan # Add missing column
    # else sp_inflation_df remains empty

    # 3. Outer Merge All Assets (S&P 500, Gold, Bonds) (for Tab 2)
    asset_ts_data = pd.DataFrame() # Initialize empty df
    all_asset_dfs = [df for df in [df_sp500_resampled, df_gold_resampled, df_bonds_resampled] if df is not None]
    if len(all_asset_dfs) > 1:
        print(f"DEBUG: Performing OUTER merge on {len(all_asset_dfs)} resampled asset DataFrames...")
        # Ensure all dataframes have the same index name before merging
        base_index_name = all_asset_dfs[0].index.name if all_asset_dfs[0].index.name else 'Date'
        for i, df in enumerate(all_asset_dfs):
            if df.index.name != base_index_name:
                print(f"WARN: Aligning index name for asset df {i} ('{df.index.name}' -> '{base_index_name}')")
                df.index.name = base_index_name
        
        # Perform the outer merge iteratively
        asset_ts_data = all_asset_dfs[0]
        for i in range(1, len(all_asset_dfs)):
            asset_ts_data = pd.merge(asset_ts_data, all_asset_dfs[i], left_index=True, right_index=True, how='outer')
        print(f"DEBUG: Combined resampled asset data shape: {asset_ts_data.shape}")
    elif len(all_asset_dfs) == 1:
        asset_ts_data = all_asset_dfs[0].copy()
        print("WARN: Only one asset DataFrame available for outer merge.")
    else:
        print("ERROR: No asset DataFrames available for outer merge.")
        # Create empty indexed DF if none loaded. Ensure index name matches expected 'DateTime' later.
        asset_ts_data = pd.DataFrame(index=pd.to_datetime([]))
        asset_ts_data.index.name = 'Date'

    # 4. Reset index to make 'Date' (or 'DateTime') a column for both dataframes
    if not sp_inflation_df.empty:
        sp_inflation_df = sp_inflation_df.reset_index()
        # Ensure the date column is named 'DateTime'
        date_col_sp = next((col for col in ['Date', 'index', 'DateTime'] if col in sp_inflation_df.columns), None)
        if date_col_sp and date_col_sp != 'DateTime':
            sp_inflation_df = sp_inflation_df.rename(columns={date_col_sp: 'DateTime'})
        elif 'DateTime' not in sp_inflation_df.columns:
            print("ERROR: Could not identify date column to rename to 'DateTime' in sp_inflation_df")
        print("DEBUG: Reset index for sp_inflation_df, 'DateTime' column ensured.")

    if not asset_ts_data.empty:
        asset_ts_data = asset_ts_data.reset_index()
        # Ensure the date column is named 'DateTime'
        date_col_asset = next((col for col in ['Date', 'index', 'DateTime'] if col in asset_ts_data.columns), None)
        if date_col_asset and date_col_asset != 'DateTime':
            asset_ts_data = asset_ts_data.rename(columns={date_col_asset: 'DateTime'})
        elif 'DateTime' not in asset_ts_data.columns:
             print("ERROR: Could not identify date column to rename to 'DateTime' in asset_ts_data")
        print("DEBUG: Reset index for asset_ts_data, 'DateTime' column ensured.")

    # --- Apply >= 1972 Filter AFTER resampling/merging --- ## USER SELECTED 1972
    filter_date = pd.Timestamp('1972-01-01')
    print(f"DEBUG: Applying final filter for dates >= {filter_date}...")

    if not sp_inflation_df.empty:
        if 'DateTime' in sp_inflation_df.columns:
            sp_inflation_df['DateTime'] = pd.to_datetime(sp_inflation_df['DateTime']) # Ensure datetime type
            original_rows_sp = len(sp_inflation_df)
            sp_inflation_df = sp_inflation_df[sp_inflation_df['DateTime'] >= filter_date].copy()
            print(f"DEBUG: Filtered S&P/Inflation data >= {filter_date}. Shape: {sp_inflation_df.shape}. (Was {original_rows_sp} rows before)")
        else:
             print("ERROR: 'DateTime' column missing before final filter for S&P/Inflation data.")
             sp_inflation_df = pd.DataFrame() # Make empty if date column is lost

    if not asset_ts_data.empty:
        if 'DateTime' in asset_ts_data.columns:
            asset_ts_data['DateTime'] = pd.to_datetime(asset_ts_data['DateTime']) # Ensure datetime type
            original_rows_asset = len(asset_ts_data)
            asset_ts_data = asset_ts_data[asset_ts_data['DateTime'] >= filter_date].copy()
            print(f"DEBUG: Filtered Asset data >= {filter_date}. Shape: {asset_ts_data.shape}. (Was {original_rows_asset} rows before)")
        else:
            print("ERROR: 'DateTime' column missing before final filter for Asset data.")
            asset_ts_data = pd.DataFrame() # Make empty if date column is lost

    # --- Final Check and Return ---
    if sp_inflation_df.empty or asset_ts_data.empty:
        print("ERROR: Both S&P/Inflation and Asset data failed to load or resample.")
        # sp_inflation_df and asset_ts_data remain empty
    else:
        print("LOG: Loaded sp_inflation_data: shape={}, min_date={}, max_date={}".format(sp_inflation_df.shape, sp_inflation_df['DateTime'].min(), sp_inflation_df['DateTime'].max()))
        print("LOG: Loaded asset_ts_data: shape={}, min_date={}, max_date={}".format(asset_ts_data.shape, asset_ts_data['DateTime'].min(), asset_ts_data['DateTime'].max()))
        print("DEBUG: load_data function finished.")
    # Return the processed S&P/Inflation df and the filtered asset df
    return sp_inflation_df.copy(), asset_ts_data.copy()

with st.spinner('Loading data...'):
    t0 = time.time()
    sp_inflation_data, asset_ts_data = load_data()
    t1 = time.time()
    print(f"DEBUG: Data loading complete. Took {t1-t0:.2f} seconds.")

# Check if dataframes are empty after loading attempt
if sp_inflation_data.empty or asset_ts_data.empty:
    # Keep st.error here for frontend visibility of critical failure
    st.error("Failed to load necessary data. Please check the data sources and try again.")
    print("ERROR: Halting execution due to empty dataframes after load.") # Add terminal log
    st.stop() # Stop execution if data loading failed

# Ensure 'DateTime' is datetime type
sp_inflation_data['DateTime'] = pd.to_datetime(sp_inflation_data['DateTime'])
asset_ts_data['DateTime'] = pd.to_datetime(asset_ts_data['DateTime'])

# Sidebar User Inputs
st.sidebar.header("User Input")
# Single moving average input for both S&P 500 and Inflation
ma_length = st.sidebar.number_input(
    "Moving Average Length (Months)",
    min_value=1,
    max_value=24,
    value=12,
    step=1,
    key='ma_length',
    help="Number of months for the moving average window for both S&P 500 and Inflation."
)
sp500_n = ma_length
inflation_n = ma_length

# Define a color palette for regimes
color_palette = [
    'green', 'yellow', 'orange', 'red', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'olive',
    'blue', 'gray', 'black', 'teal', 'navy', 'maroon'
]

# Regime color mapping for background shading
regime_bg_colors = {
    1: 'rgba(0, 128, 255, 0.13)',   # Rising Growth, Rising Inflation - Light Blue
    2: 'rgba(0, 255, 0, 0.13)',     # Rising Growth, Falling Inflation - Light Green
    3: 'rgba(255, 0, 0, 0.13)',     # Falling Growth, Rising Inflation - Light Red
    4: 'rgba(255, 255, 0, 0.13)'    # Falling Growth, Falling Inflation - Light Yellow
}

# Regime color mapping for legend (match background, but higher opacity)
regime_legend_colors = {
    1: 'rgba(0, 128, 255, 0.7)',   # Light Blue, more visible
    2: 'rgba(0, 255, 0, 0.7)',    # Light Green
    3: 'rgba(255, 0, 0, 0.7)',    # Light Red
    4: 'rgba(255, 255, 0, 0.7)'   # Light Yellow
}

# Asset and macro color mapping
asset_colors = {
    'S&P 500': 'blue',
    'Gold': 'gold',
    'Bonds': 'black',
    'Inflation Rate': 'red',
    'S&P 500 MA': 'blue',
    'Inflation Rate MA': 'red',
}

# Caching dynamic computations
@st.cache_data
def compute_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

@st.cache_data
def compute_growth(ma_data):
    return ma_data.diff()

# --- Logging for Moving Average Computation ---
t0 = time.time()
with st.spinner('Computing Moving Averages...'):
    sp_inflation_data['S&P 500 MA'] = compute_moving_average(
        sp_inflation_data['S&P 500'], window_size=sp500_n
    )
    sp_inflation_data['Inflation Rate MA'] = compute_moving_average(
        sp_inflation_data['Inflation Rate'], window_size=inflation_n
    )
    print("DEBUG: Moving averages computed.")
t1 = time.time()
print(f"DEBUG: Moving average computation took {t1-t0:.2f} seconds.")

# --- Logging for Growth Computation ---
t0 = time.time()
with st.spinner('Computing Growth...'):
    sp_inflation_data['S&P 500 MA Growth'] = compute_growth(
        sp_inflation_data['S&P 500 MA']
    )
    sp_inflation_data['Inflation Rate MA Growth'] = compute_growth(
        sp_inflation_data['Inflation Rate MA']
    )
    print("DEBUG: Growth computed.")
t1 = time.time()
print(f"DEBUG: Growth computation took {t1-t0:.2f} seconds.")

# Now that we have the growth, we can get min and max values
sp500_growth = sp_inflation_data['S&P 500 MA Growth'].dropna()
inflation_growth = sp_inflation_data['Inflation Rate MA Growth'].dropna()

sp500_min = float(sp500_growth.min())
sp500_max = float(sp500_growth.max())
inflation_min = float(inflation_growth.min())
inflation_max = float(inflation_growth.max())

# Elegant regime definitions: Only check for greater/smaller than 0 for both growth rates
regime_definitions = [
    {
        'Regime': 1,
        'S&P 500 Lower': 0,
        'S&P 500 Upper': float('inf'),
        'Inflation Lower': 0,
        'Inflation Upper': float('inf'),
        'Label': 'Rising Growth & Rising Inflation'
    },
    {
        'Regime': 2,
        'S&P 500 Lower': 0,
        'S&P 500 Upper': float('inf'),
        'Inflation Lower': float('-inf'),
        'Inflation Upper': 0,
        'Label': 'Rising Growth & Falling Inflation'
    },
    {
        'Regime': 3,
        'S&P 500 Lower': float('-inf'),
        'S&P 500 Upper': 0,
        'Inflation Lower': 0,
        'Inflation Upper': float('inf'),
        'Label': 'Falling Growth & Rising Inflation'
    },
    {
        'Regime': 4,
        'S&P 500 Lower': float('-inf'),
        'S&P 500 Upper': 0,
        'Inflation Lower': float('-inf'),
        'Inflation Upper': 0,
        'Label': 'Falling Growth & Falling Inflation'
    }
]
# Assign colors and labels to regimes
regime_colors = {1: 'green', 2: 'yellow', 3: 'red', 4: 'blue'}
regime_labels_dict = {
    1: 'Rising Growth & Rising Inflation',
    2: 'Rising Growth & Falling Inflation',
    3: 'Falling Growth & Rising Inflation',
    4: 'Falling Growth & Falling Inflation',
    'Unknown': 'Unknown'
}

# Function to assign regimes based on thresholds
@st.cache_data
def assign_regimes(sp_inflation_df, regime_definitions):
    # Initialize Regime column
    sp_inflation_df['Regime'] = np.nan

    # Iterate over regimes and assign regime numbers
    for regime in regime_definitions:
        mask = (
            (sp_inflation_df['S&P 500 MA Growth'] >= regime['S&P 500 Lower']) &
            (sp_inflation_df['S&P 500 MA Growth'] < regime['S&P 500 Upper']) &
            (sp_inflation_df['Inflation Rate MA Growth'] >= regime['Inflation Lower']) &
            (sp_inflation_df['Inflation Rate MA Growth'] < regime['Inflation Upper'])
        )
        sp_inflation_df.loc[mask, 'Regime'] = regime['Regime']
    return sp_inflation_df

# --- Logging for Regime Assignment ---
t0 = time.time()
with st.spinner('Assigning Regimes...'):
    sp_inflation_data = assign_regimes(sp_inflation_data, regime_definitions)
    print("DEBUG: Regimes assigned.")
t1 = time.time()
print(f"DEBUG: Regime assignment took {t1-t0:.2f} seconds.")

# Handle any NaN regimes (should not happen)
sp_inflation_data['Regime'] = sp_inflation_data['Regime'].fillna('Unknown')
if 'Unknown' in sp_inflation_data['Regime'].unique():
    regime_colors['Unknown'] = 'lightgrey'
    regime_labels_dict['Unknown'] = 'Unknown'

# --- Logging for Tab Rendering ---
t0 = time.time()
print("DEBUG: Starting Tab rendering.")
tabs = st.tabs(["Regime Visualization", "Asset Performance & Metrics"])
t1 = time.time()
print(f"DEBUG: Tab setup took {t1-t0:.2f} seconds.")

# Tab 1: Regime Visualization
with tabs[0]:
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Visualization</h2>", unsafe_allow_html=True)
    t_tab1 = time.time()
    print("DEBUG: Rendering Tab 1: Regime Visualization.")
    # Hardcoded settings for chart 1 (enable all except log scales)
    show_sp500_ma = True
    show_inflation_ma = True
    show_sp500 = True
    show_inflation = True
    log_scale_sp500 = False
    log_scale_inflation_rate = False
    
    # Initialize the plot
    fig = go.Figure()
    print("DEBUG: Tab 1 - Initialized go.Figure.") # Added debug print

    # Add shaded regions for regimes (updated to handle continuous periods)
    # Identify where the regime changes
    sp_inflation_data['Regime_Change'] = (sp_inflation_data['Regime'] != sp_inflation_data['Regime'].shift()).cumsum()
    print("DEBUG: Tab 1 - Calculated Regime_Change.") # Added debug print

    # Group by 'Regime' and 'Regime_Change' to get continuous periods
    grouped = sp_inflation_data.groupby(['Regime', 'Regime_Change'])
    print(f"DEBUG: Tab 1 - Grouped regimes. Number of groups: {len(grouped)}") # Added debug print

    # Collect regime periods
    regime_periods = []
    print("DEBUG: Tab 1 - Starting regime period collection loop.") # Added debug print
    loop_count = 0
    for (regime, _), group in grouped:
        color = regime_bg_colors.get(regime, 'rgba(200,200,200,0.10)')
        start_date_regime = group['DateTime'].iloc[0]
        end_date_regime = group['DateTime'].iloc[-1]
        regime_periods.append({
            'Regime': regime,
            'Start Date': start_date_regime,
            'End Date': end_date_regime
        })
        loop_count += 1
        if loop_count % 50 == 0: # Print progress every 50 groups
             print(f"DEBUG: Tab 1 - Processed {loop_count} regime groups.")
    print(f"DEBUG: Tab 1 - Finished regime period collection loop. Total periods collected: {len(regime_periods)}") # Added debug print

    # Sort regime periods by start date
    regime_periods_df = pd.DataFrame(regime_periods)
    regime_periods_df = regime_periods_df.sort_values('Start Date').reset_index(drop=True)
    print("DEBUG: Tab 1 - Sorted regime periods DataFrame.") # Added debug print

    # Adjust end dates and add vrects
    print("DEBUG: Tab 1 - Starting add_vrect loop.") # Added debug print
    vrect_count = 0
    for i in range(len(regime_periods_df)):
        start_date_regime = regime_periods_df.loc[i, 'Start Date']
        regime = regime_periods_df.loc[i, 'Regime']
        color = regime_bg_colors.get(regime, 'rgba(200,200,200,0.10)')
        if i < len(regime_periods_df) - 1:
            # Set end date to the exact same day as the next regime's start date
            end_date_regime = regime_periods_df.loc[i+1, 'Start Date']
        else:
            # For the last regime, set end date to the maximum date
            end_date_regime = sp_inflation_data['DateTime'].max()
        # Ensure end_date_regime is not before start_date_regime
        if end_date_regime < start_date_regime:
            end_date_regime = start_date_regime
        # Add vrect for this regime
        fig.add_vrect(
            x0=start_date_regime,
            x1=end_date_regime,
            fillcolor=color,
            opacity=1.0,
            layer="below",
            line_width=0
        )
        vrect_count += 1
        if vrect_count % 50 == 0: # Print progress every 50 vrects
            print(f"DEBUG: Tab 1 - Added {vrect_count} vrects.")
    print(f"DEBUG: Tab 1 - Finished add_vrect loop. Total vrects added: {vrect_count}") # Added debug print

    # --- Optimization: Create customdata ONCE ---
    print("DEBUG: Tab 1 - Preparing customdata array...") # Added debug print
    try:
        # Ensure required columns exist and handle potential NaNs before stacking
        required_cols = ['Regime', 'S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']
        # Check if all required columns are present
        if not all(col in sp_inflation_data.columns for col in required_cols):
             raise ValueError(f"Missing one or more required columns for customdata: {required_cols}")

        # Map regimes (handle potential missing keys in dict gracefully)
        regime_labels = sp_inflation_data['Regime'].map(lambda x: regime_labels_dict.get(x, 'Unknown'))
        sp500_data = sp_inflation_data['S&P 500'].fillna(0)
        sp500_ma_data = sp_inflation_data['S&P 500 MA'].fillna(0)
        inflation_data = sp_inflation_data['Inflation Rate'].fillna(0)
        inflation_ma_data = sp_inflation_data['Inflation Rate MA'].fillna(0)
        customdata = np.stack((sp500_data, sp500_ma_data, inflation_data, inflation_ma_data, regime_labels), axis=-1)
        print(f"DEBUG: Tab 1 - Customdata array created successfully. Shape: {customdata.shape}") # Added debug print
    except Exception as e:
        print(f"ERROR: Tab 1 - Failed to create customdata array: {e}")
        st.error(f"Failed to prepare data for plotting: {e}")
        # Assign a dummy array or stop execution if customdata is critical
        customdata = np.empty((len(sp_inflation_data), 5)) # Example dummy
        # Or potentially st.stop() here if the plot can't proceed

    # Add traces based on user selection, reusing the customdata array
    print("DEBUG: Tab 1 - Starting add_trace section.") # Added debug print
    hover_template_sp500 = (
        'S&P 500: %{customdata[0]:.2f}<br>'
        'Regime: %{customdata[4]}<extra></extra>'
    )
    hover_template_sp500_ma = (
        'S&P 500 MA: %{customdata[1]:.2f}<br>'
        'Regime: %{customdata[4]}<extra></extra>'
    )
    hover_template_inflation = (
        'Inflation Rate: %{customdata[2]:.2f}<br>'
        'Regime: %{customdata[4]}<extra></extra>'
    )
    hover_template_inflation_ma = (
        'Inflation Rate MA: %{customdata[3]:.2f}<br>'
        'Regime: %{customdata[4]}<extra></extra>'
    )
    if show_sp500:
        print("DEBUG: Tab 1 - Preparing S&P 500 trace...")
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500'],
            mode='lines',
            name='S&P 500',
            line=dict(color=asset_colors['S&P 500'], dash='dot'),
            yaxis='y1',
            customdata=customdata,
            hovertemplate=hover_template_sp500
        ))
        print("DEBUG: Tab 1 - Added S&P 500 trace.")
    if show_sp500_ma:
        print("DEBUG: Tab 1 - Preparing S&P 500 MA trace...")
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500 MA'],
            mode='lines',
            name=f'S&P 500 MA ({sp500_n}m)',
            line=dict(color=asset_colors['S&P 500 MA']),
            yaxis='y1',
            customdata=customdata,
            hovertemplate=hover_template_sp500_ma
        ))
        print("DEBUG: Tab 1 - Added S&P 500 MA trace.")
    if show_inflation:
        print("DEBUG: Tab 1 - Preparing Inflation Rate trace...")
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate'],
            mode='lines',
            name='Inflation Rate',
            line=dict(color=asset_colors['Inflation Rate'], dash='dot'),
            yaxis='y2',
            customdata=customdata,
            hovertemplate=hover_template_inflation
        ))
        print("DEBUG: Tab 1 - Added Inflation Rate trace.")
    if show_inflation_ma:
        print("DEBUG: Tab 1 - Preparing Inflation Rate MA trace...")
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate MA'],
            mode='lines',
            name=f'Inflation Rate MA ({inflation_n}m)',
            line=dict(color=asset_colors['Inflation Rate MA']),
            yaxis='y2',
            customdata=customdata,
            hovertemplate=hover_template_inflation_ma
        ))
        print("DEBUG: Tab 1 - Added Inflation Rate MA trace.")
    print("DEBUG: Tab 1 - Finished add_trace section.")

    # Update layout with optional log scales
    print("DEBUG: Tab 1 - Updating layout...") # Added debug print
    fig.update_layout(
        title={
            'text': 'Macro Regime Timeline: S&P 500 & Inflation',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=dict(
                text='S&P 500',
                font=dict(
                    family='Arial',
                    size=14,
                    color='black'
                )
            ),
            tickfont=dict(
                family='Arial',
                size=12,
                color='black'
            ),
            side='left',
            type='log' if log_scale_sp500 else 'linear'
        ),
        yaxis2=dict(
            title=dict(
                text='Inflation Rate',
                font=dict(
                    family='Arial',
                    size=14,
                    color='black'
                )
            ),
            tickfont=dict(
                family='Arial',
                size=12,
                color='black'
            ),
            overlaying='y',
            side='right',
            type='log' if log_scale_inflation_rate else 'linear'
        ),
        hovermode='x unified',
        width=1200,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="black"
        ),
        showlegend=False
    )
    print("DEBUG: Tab 1 - Layout updated.") # Added debug print

    print("LOG: Chart data: shape={}, min_date={}, max_date={}".format(sp_inflation_data.shape, sp_inflation_data['DateTime'].min(), sp_inflation_data['DateTime'].max()))
    # Display the plot
    print("DEBUG: Tab 1 - Calling st.plotly_chart(fig)...") # Added debug print
    st.plotly_chart(fig, use_container_width=False)
    print("DEBUG: Tab 1 Plotting complete.") # Added debug print
    
    # Create Regime Legend under the graph with regime definitions
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Legend</h2>", unsafe_allow_html=True)
    regime_legend_html = "<ul>"
    
    # Add numeric regimes
    for regime_num in sorted(regime_labels_dict.keys(), key=lambda x: int(x) if x != 'Unknown' else float('inf')):
        if regime_num == 'Unknown':
            continue
        color = regime_legend_colors.get(regime_num, 'grey')
        label = regime_labels_dict.get(regime_num, 'Unknown')
        regime_legend_html += f"<li><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px; border-radius:3px; border:1px solid #888;'></span> <b>{label}</b></li>"
    regime_legend_html += "</ul>"
    st.markdown(regime_legend_html, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Scatter Plots</h2>", unsafe_allow_html=True)
    
    # --- First Scatter Plot ---
    # Prepare data for plotting
    derivative_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna()
    derivative_df = derivative_df.sort_values('DateTime').reset_index(drop=True)
    n_points = len(derivative_df)
    window_size = 50
    # Show only the latest 50 data points for scatterplot
    window_df = derivative_df.iloc[-window_size:].copy()
    # Ensure required columns exist for customdata
    for col in ['S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']:
        if col not in window_df.columns:
            window_df[col] = np.nan
    # Get the actual date span for the title
    if not window_df.empty:
        date_start = window_df['DateTime'].iloc[0].strftime('%Y-%m-%d')
        date_end = window_df['DateTime'].iloc[-1].strftime('%Y-%m-%d')
        date_span = f"{date_start} to {date_end}"
    else:
        date_span = "N/A"
    # Black to yellow colormap for gradient
    from matplotlib.colors import LinearSegmentedColormap
    by_cmap = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=window_size)
    colors = [mcolors.to_hex(by_cmap(i/(window_size-1))) for i in range(len(window_df))]
    scatter_fig = go.Figure()
    # Determine axis ranges for quadrants (for scatter_fig)
    x_min = float(window_df['S&P 500 MA Growth'].min()) if not window_df.empty else -1
    x_max = float(window_df['S&P 500 MA Growth'].max()) if not window_df.empty else 1
    y_min = float(window_df['Inflation Rate MA Growth'].min()) if not window_df.empty else -1
    y_max = float(window_df['Inflation Rate MA Growth'].max()) if not window_df.empty else 1
    # Ensure zero is inside the plot for quadrant clarity
    # Expand the axis ranges for quadrant backgrounds to allow backgrounds to show when zoomed out
    x_margin = 0.1 * (x_max - x_min) if x_max != x_min else 0.1
    y_margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.1
    x_bg_min = min(x_min, 0) - x_margin
    x_bg_max = max(x_max, 0) + x_margin
    y_bg_min = min(y_min, 0) - y_margin
    y_bg_max = max(y_max, 0) + y_margin
    x_range = [x_bg_min, x_bg_max]
    y_range = [y_bg_min, y_bg_max]
    # Add quadrant backgrounds to scatter_fig using expanded ranges
    scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max, y0=0, y1=y_bg_max,
        fillcolor=regime_bg_colors[1],
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max, y0=y_bg_min, y1=0,
        fillcolor=regime_bg_colors[2],
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min, x1=0, y0=0, y1=y_bg_max,
        fillcolor=regime_bg_colors[3],
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min, x1=0, y0=y_bg_min, y1=0,
        fillcolor=regime_bg_colors[4],
        line_width=0, layer="below"
    )
    scatter_fig.add_trace(go.Scatter(
        x=window_df['S&P 500 MA Growth'],
        y=window_df['Inflation Rate MA Growth'],
        mode='lines+markers',
        marker=dict(
            color=colors,
            size=12,
            line=dict(width=1, color='black')
        ),
        line=dict(color='#444444', width=2, dash='solid'),
        text=window_df['DateTime'].dt.strftime('%Y-%m-%d'),
        customdata=np.stack([
            window_df['S&P 500'],
            window_df['S&P 500 MA'],
            window_df['Inflation Rate'],
            window_df['Inflation Rate MA'],
        ], axis=-1),
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500: %{customdata[0]:.2f}<br>' +
            'S&P 500 MA: %{customdata[1]:.2f}<br>' +
            'Inflation Rate: %{customdata[2]:.2f}<br>' +
            'Inflation Rate MA: %{customdata[3]:.2f}<extra></extra>'
        ),
        name=date_span
    ))
    scatter_fig.update_layout(
        xaxis_title='S&P 500 MA Growth',
        yaxis_title='Inflation Rate MA',
        title={
            'text': f'Regime Movement Over Time ({date_span})',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        width=800,
        height=600,
        showlegend=False,
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range)
    )
    st.plotly_chart(scatter_fig)
    print("DEBUG: Tab 1 Scatter Plot complete.")
    
    # --- Second Scatter Plot ---
    # Prepare data for plotting
    all_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna().sort_values('DateTime')
    # Ensure required columns exist for customdata
    for col in ['S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']:
        if col not in all_df.columns:
            all_df.loc[:, col] = np.nan
    N_all = len(all_df)
    by_cmap_all = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=N_all)
    all_colors = [mcolors.to_hex(by_cmap_all(i/(N_all-1))) for i in range(N_all)]
    all_scatter_fig = go.Figure()
    # Determine axis ranges for quadrants (for all_scatter_fig)
    x_min_all = float(all_df['S&P 500 MA Growth'].min()) if not all_df.empty else -1
    x_max_all = float(all_df['S&P 500 MA Growth'].max()) if not all_df.empty else 1
    y_min_all = float(all_df['Inflation Rate MA Growth'].min()) if not all_df.empty else -1
    y_max_all = float(all_df['Inflation Rate MA Growth'].max()) if not all_df.empty else 1
    # Expand the axis ranges for quadrant backgrounds to allow backgrounds to show when zoomed out
    x_margin_all = 0.1 * (x_max_all - x_min_all) if x_max_all != x_min_all else 0.1
    y_margin_all = 0.1 * (y_max_all - y_min_all) if y_max_all != y_min_all else 0.1
    x_bg_min_all = min(x_min_all, 0) - x_margin_all
    x_bg_max_all = max(x_max_all, 0) + x_margin_all
    y_bg_min_all = min(y_min_all, 0) - y_margin_all
    y_bg_max_all = max(y_max_all, 0) + y_margin_all
    x_range_all = [x_bg_min_all, x_bg_max_all]
    y_range_all = [y_bg_min_all, y_bg_max_all]
    all_scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max_all, y0=0, y1=y_bg_max_all,
        fillcolor=regime_bg_colors[1],
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max_all, y0=y_bg_min_all, y1=0,
        fillcolor=regime_bg_colors[2],
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min_all, x1=0, y0=0, y1=y_bg_max_all,
        fillcolor=regime_bg_colors[3],
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min_all, x1=0, y0=y_bg_min_all, y1=0,
        fillcolor=regime_bg_colors[4],
        line_width=0, layer="below"
    )
    all_scatter_fig.add_trace(go.Scatter(
        x=all_df['S&P 500 MA Growth'],
        y=all_df['Inflation Rate MA Growth'],
        mode='markers',
        marker=dict(
            color=all_colors,
            size=10,
            line=dict(width=1, color='black')
        ),
        text=all_df['DateTime'].dt.strftime('%Y-%m-%d'),
        customdata=np.stack([
            all_df['S&P 500'],
            all_df['S&P 500 MA'],
            all_df['Inflation Rate'],
            all_df['Inflation Rate MA'],
        ], axis=-1),
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500: %{customdata[0]:.2f}<br>' +
            'S&P 500 MA: %{customdata[1]:.2f}<br>' +
            'Inflation Rate: %{customdata[2]:.2f}<br>' +
            'Inflation Rate MA: %{customdata[3]:.2f}<extra></extra>'
        ),
        name='All Data (Oldest to Newest)'
    ))
    all_scatter_fig.update_layout(
        xaxis_title='S&P 500 MA Growth',
        yaxis_title='Inflation Rate MA',
        title={
            'text': 'Scatter Plot of All Regimes',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        width=800,
        height=600,
        showlegend=False,
        xaxis=dict(range=x_range_all),
        yaxis=dict(range=y_range_all)
    )
    st.plotly_chart(all_scatter_fig)
    print("DEBUG: Tab 1 All-Data Scatter Plot complete.")

    # --- DATA SOURCES SECTION ---
    st.markdown("""
---
<h2>Data Sources</h2>
<ul>
  <li>Raw Data URLs Used in This App:
    <ul>
      <li>S&amp;P 500 Data: <a href="https://www.longtermtrends.net/data-sp500-since-1871/" target="_blank">longtermtrends.net/data-sp500-since-1871/</a></li>
      <li>Inflation Rate Data: <a href="https://www.longtermtrends.net/data-inflation-forecast/" target="_blank">longtermtrends.net/data-inflation-forecast/</a></li>
      <li>Total Return Bond Index Data: <a href="https://www.longtermtrends.net/data-total-return-bond-index/" target="_blank">longtermtrends.net/data-total-return-bond-index/</a></li>
      <li>Gold Price Data: <a href="https://www.longtermtrends.net/data-gold-since-1792/" target="_blank">longtermtrends.net/data-gold-since-1792/</a></li>
    </ul>
  </li>
  <li>S&amp;P 500:
    <ul>
      <li>Recent Prices: <a href="https://fred.stlouisfed.org/series/SP500" target="_blank">FRED S&amp;P 500</a></li>
      <li>From 1928 until 2023: <a href="https://finance.yahoo.com/quote/%5ESPX/history/" target="_blank">Yahoo Finance S&amp;P 500</a></li>
      <li>Until 1927: <a href="https://www.multpl.com/s-p-500-historical-prices/table/by-month" target="_blank">Multpl S&amp;P 500</a></li>
    </ul>
  </li>
  <li>Total Return Bond Index:
    <ul>
      <li>Since 1973: <a href="https://fred.stlouisfed.org/series/BAMLCC0A0CMTRIV" target="_blank">FRED BAMLCC0A0CMTRIV</a></li>
      <li>Until 1973: <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3805927" target="_blank">McQuarrie: Where Siegel Went Awry: Outdated Sources &amp; Incomplete Data</a></li>
    </ul>
  </li>
  <li>Gold:
    <ul>
      <li>Since 2000: <a href="https://finance.yahoo.com/quote/GC=F/" target="_blank">Yahoo Finance: COMEX Gold</a></li>
      <li>Until 2000: <a href="https://onlygold.com/gold-prices/historical-gold-prices/" target="_blank">OnlyGold: Historical Prices</a></li>
    </ul>
  </li>
  <li>Inflation:
    <ul>
      <li>Latest data point: <a href="https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting" target="_blank">Cleveland Fed: Inflation Nowcasting</a></li>
      <li>Since 1913: <a href="https://fred.stlouisfed.org/series/CPIAUCNS" target="_blank">FRED: CPI</a></li>
      <li>Until 1913: <a href="http://www.econ.yale.edu/~shiller/data.htm" target="_blank">Yale University: Online Data Robert Shiller</a></li>
    </ul>
  </li>
</ul>
""", unsafe_allow_html=True)
    t_tab1_end = time.time()
    print(f"DEBUG: Tab 1 rendering took {t_tab1_end-t_tab1:.2f} seconds.")

# Tab 2: Asset Performance & Metrics
with tabs[1]:
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Asset Performance Over Time</h2>", unsafe_allow_html=True)
    t_tab2 = time.time()
    print("DEBUG: Rendering Tab 2: Asset Performance & Metrics (MERGED).")
    log_scale_normalized = False

    # Merge asset data with regime assignments
    t_merge_start = time.time()
    with st.spinner('Merging asset data with regimes...'):
        merged_asset_data = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
        print(f"DEBUG: Tab 2 - Asset data merged. Shape: {merged_asset_data.shape}")
    t_merge_end = time.time()
    print(f"DEBUG: Tab 2 - Asset merge took {t_merge_end-t_merge_start:.3f} seconds.")

    # Initialize the plot
    t_fig_start = time.time()
    fig2 = go.Figure()
    t_fig_init = time.time()
    print(f"DEBUG: Tab 2 - go.Figure() init took {t_fig_init-t_fig_start:.3f} seconds.")

    # Add shaded regions for regimes (optimized: bulk shape update)
    t_shapes_start = time.time()
    shapes = []
    for i in range(len(regime_periods_df)):
        start_date_regime = regime_periods_df.loc[i, 'Start Date']
        regime = regime_periods_df.loc[i, 'Regime']
        color = regime_bg_colors.get(regime, 'rgba(200,200,200,0.10)')
        if i < len(regime_periods_df) - 1:
            # Set end date to the exact same day as the next regime's start date
            end_date_regime = regime_periods_df.loc[i+1, 'Start Date']
        else:
            # For the last regime, set end date to the maximum date
            end_date_regime = merged_asset_data['DateTime'].max()
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_date_regime,
            x1=end_date_regime,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=1.0,
            layer="below",
            line_width=0
        ))
        if (i % 10 == 0) or (i == len(regime_periods_df)-1):
            print(f"DEBUG: Tab 2 - shape {i+1}/{len(regime_periods_df)} prepared.")
    t_shapes_end = time.time()
    print(f"DEBUG: Tab 2 - Regime shapes preparation loop took {t_shapes_end-t_shapes_start:.3f} seconds.")

    t_vrect_start = time.time()
    fig2.update_layout(shapes=shapes,
        title={
            'text': 'Asset Performance Over Time (Normalized to 100 at First Available Date)',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Normalized Price',
            type='linear'
        ),
        hovermode='x unified',
        width=1200,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="black"
        ),
    )
    t_vrect_end = time.time()
    print(f"DEBUG: Tab 2 - Regime vrects bulk update took {t_vrect_end-t_vrect_start:.3f} seconds.")

    # Plot each asset
    t_plot_start = time.time()
    for asset in ['S&P 500', 'Gold', 'Bonds']:
        asset_data = merged_asset_data[['DateTime', asset, 'Regime']].copy()
        asset_data = asset_data[asset_data['Regime'] != 'Unknown'].copy()
        asset_data = asset_data.dropna(subset=[asset, 'Regime']).copy()
        asset_data['Regime Label'] = asset_data['Regime'].map(regime_labels_dict)
        if asset_data.empty:
            st.warning(f"No data available for asset {asset} in the selected date range.")
            continue
        asset_data['Actual Price'] = asset_data[asset]
        asset_data['Normalized Price'] = 100 * asset_data['Actual Price'] / asset_data['Actual Price'].iloc[0]
        customdata = np.stack([
            asset_data['Actual Price'],
            asset_data['Regime Label']
        ], axis=-1)
        price_label = asset
        hovertemplate = (
            "Date: %{x|%Y-%m-%d}<br>"
            "Regime: %{customdata[1]}<br>"
            "Normalized Price: %{y:.2f}<br>"
            + price_label + ": %{customdata[0]:.2f}<extra></extra>"
        )
        t_add_trace_start = time.time()
        fig2.add_trace(go.Scatter(
            x=asset_data['DateTime'],
            y=asset_data['Normalized Price'],
            mode='lines',
            name=asset,
            line=dict(color=asset_colors.get(asset, 'gray'), width=2),
            customdata=customdata,
            connectgaps=False,
            hovertemplate=hovertemplate
        ))
        t_add_trace_end = time.time()
        print(f"DEBUG: Tab 2 - Asset {asset} trace add took {t_add_trace_end-t_add_trace_start:.3f} seconds.")
    t_plot_end = time.time()
    print(f"DEBUG: Tab 2 - Asset plotting loop took {t_plot_end-t_plot_start:.3f} seconds.")

    # Remove all download buttons except for the last table CSV
    st.plotly_chart(fig2, use_container_width=False)

    # --- Performance Metrics per Regime (from former Tab 3) ---
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Trade Log</h2>", unsafe_allow_html=True)
    print("DEBUG: Rendering Performance Metrics Section (from former Tab 3).")
    t_metrics_start = time.time()
    with st.spinner('Merging asset data with regimes...'):
        merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
        print(f"DEBUG: Tab 2 - Asset data merged (for metrics). Shape: {merged_asset_data_metrics.shape}")
    trade_log_results = [] # New list for detailed trade log
    t_metrics_loop_start = time.time()
    for asset in ['S&P 500', 'Gold', 'Bonds']:
        asset_data = merged_asset_data_metrics[['DateTime', asset, 'Regime']].copy()
        asset_data = asset_data[asset_data['Regime'] != 'Unknown'].copy()
        asset_data = asset_data.dropna(subset=[asset, 'Regime']).copy()
        asset_data['Regime Label'] = asset_data['Regime'].map(regime_labels_dict)
        if asset_data.empty:
            st.warning(f"No data available for asset {asset} in the selected date range.")
            continue
        # --- Simulate buy at regime change, sell at next regime change ---
        asset_data = asset_data.sort_values('DateTime').reset_index(drop=True)
        # Use shift(-1) to check the *next* row's regime to define the end of a period
        # asset_data['Regime_End'] = (asset_data['Regime'] != asset_data['Regime'].shift(-1))
        # Group by the start of each regime change (detects contiguous blocks)
        asset_data['Regime_Start_Group'] = (asset_data['Regime'] != asset_data['Regime'].shift()).cumsum()

        # Iterate through each identified regime period group
        for group_id, group in asset_data.groupby('Regime_Start_Group', observed=False):
            if len(group) < 1: # Should not happen with cumsum logic, but safety check
                 continue

            regime_num = group['Regime'].iloc[0] # Regime number for this period
            regime_label = group['Regime Label'].iloc[0] # Regime label for this period

            # Get start and end details for the period
            start_date = group['DateTime'].iloc[0]
            end_date = group['DateTime'].iloc[-1]
            price_start = group[asset].iloc[0]
            price_end = group[asset].iloc[-1]

            # Calculate period return
            period_return = (price_end - price_start) / price_start if price_start != 0 and not np.isnan(price_start) else np.nan

            # --- Store data for the detailed trade log table ---
            trade_log_results.append({
                'Asset': asset,
                'Regime': regime_label,
                'Start Date': start_date.strftime('%Y-%m-%d'), # Format date for readability
                'End Date': end_date.strftime('%Y-%m-%d'),   # Format date for readability
                'Start Price': price_start,
                'End Price': price_end,
                'Period Return': period_return
            })

            # --- Original Aggregated Performance Metrics Calculation (Needs Refactoring) ---
            # The logic below still appends one row per regime period, leading to the 325 rows.
            # We will refactor this part next to aggregate correctly.
            # For now, just get the trade log working.

            t_metrics_inner_start = time.time() # Keep timing for consistency
            # Temporary placeholder for aggregated metrics logic (will be replaced)
            if len(group) >= 2:
                returns = group[asset].pct_change().dropna()
                volatility = returns.std() * np.sqrt(12) if not returns.empty else np.nan
                cumulative = (1 + returns).cumprod()
                cumulative_max = cumulative.cummax()
                drawdown = cumulative / cumulative_max - 1
                max_drawdown = drawdown.min() if not drawdown.empty else np.nan
                sharpe_ratio = period_return / volatility if volatility != 0 and not np.isnan(volatility) and not np.isnan(period_return) else np.nan

                # performance_results.append({
                #     'Asset': asset,
                #     'Regime': regime_label,
                #     'Average Return': period_return, # Still using period return here temporarily
                #     'Volatility': volatility,
                #     'Sharpe Ratio': sharpe_ratio,
                #     'Max Drawdown': max_drawdown,
                #     # Add Start/End Date to link back if needed, or use Regime_Start_Group
                #     'Regime_Start_Group': group_id # Keep track of which period this belongs to
                # })
                t_metrics_inner_end = time.time()
                print(f"DEBUG: Tab 2 - Asset {asset} regime {regime_num} (Group {group_id}) metrics calc took {t_metrics_inner_end-t_metrics_inner_start:.3f} seconds.")
            else: # Handle single-row groups if they occur (though less likely now)
                 # performance_results.append({
                 #     'Asset': asset,
                 #     'Regime': regime_label,
                 #     'Average Return': np.nan,
                 #     'Volatility': np.nan,
                 #     'Sharpe Ratio': np.nan,
                 #     'Max Drawdown': np.nan,
                 #     'Regime_Start_Group': group_id
                 # })
                 t_metrics_inner_end = time.time()
                 print(f"DEBUG: Tab 2 - Asset {asset} regime {regime_num} (Group {group_id}) (SKIP < 2 rows) metrics calc took {t_metrics_inner_end-t_metrics_inner_start:.3f} seconds.")
            # --- End of Temporary Placeholder ---


    t_metrics_loop_end = time.time()
    print(f"DEBUG: Tab 2 - Performance metrics asset/loop took {t_metrics_loop_end-t_metrics_loop_start:.3f} seconds.")
    t_metrics_end = time.time()
    print(f"DEBUG: Tab 2 - Performance metrics computation took {t_metrics_end-t_metrics_start:.3f} seconds.")

    # --- After collecting trade_log_results, fix regime period end dates so each ends with the exact start date of the next period ---
    if trade_log_results:
        # Build a lookup for asset prices by asset and date
        asset_price_lookup = {}
        for asset in ['S&P 500', 'Bonds', 'Gold']:
            asset_price_lookup[asset] = {}
            # Use merged_asset_data_metrics to get all available prices
            for idx, row in merged_asset_data_metrics[['DateTime', asset]].dropna().iterrows():
                asset_price_lookup[asset][pd.to_datetime(row['DateTime']).strftime('%Y-%m-%d')] = row[asset]

        corrected_trade_log = []
        for asset in ['S&P 500', 'Bonds', 'Gold']:
            # Filter for this asset and sort by start date ascending
            asset_trades = [row for row in trade_log_results if row['Asset'] == asset]
            asset_trades = sorted(asset_trades, key=lambda x: pd.to_datetime(x['Start Date']))
            for i, row in enumerate(asset_trades):
                row = row.copy()  # Avoid mutating original
                if i < len(asset_trades) - 1:
                    # Set end date to the exact same day as the next start date
                    next_start = pd.to_datetime(asset_trades[i+1]['Start Date'])
                    end_date_str = next_start.strftime('%Y-%m-%d')
                    row['End Date'] = end_date_str
                    # Look up the correct end price
                    end_price = asset_price_lookup[asset].get(end_date_str, row['End Price'])
                    row['End Price'] = end_price
                    # Recompute period return
                    price_start = row['Start Price']
                    row['Period Return'] = (end_price - price_start) / price_start if price_start != 0 and not pd.isna(price_start) and not pd.isna(end_price) else float('nan')
                # else: keep original end date and end price
                corrected_trade_log.append(row)
        trade_log_results = corrected_trade_log

    # --- Create DataFrames ---
    # Aggregated Performance (still needs proper aggregation logic)
    if trade_log_results:
        trade_log_df = pd.DataFrame(trade_log_results)
        if not trade_log_df.empty:
            # --- SORTING LOGIC: Newest regime first, oldest last ---
            trade_log_df['Start Date'] = pd.to_datetime(trade_log_df['Start Date'])
            trade_log_df['End Date'] = pd.to_datetime(trade_log_df['End Date'])
            # Sort by End Date DESC, Start Date DESC, Asset order
            asset_order = ['S&P 500', 'Bonds', 'Gold']
            trade_log_df['Asset'] = pd.Categorical(trade_log_df['Asset'], categories=asset_order, ordered=True)
            trade_log_df = trade_log_df.sort_values(['End Date', 'Start Date', 'Asset'], ascending=[False, False, True])

            # --- COLORING LOGIC (define before both tables) ---
            def highlight_regime(row):
                regime_label = row['Regime']
                regime_num = None
                for k, v in regime_labels_dict.items():
                    if v == regime_label:
                        regime_num = k
                        break
                color = regime_bg_colors.get(regime_num, '#eeeeee')
                return [f'background-color: {color}'] * len(row)
            highlight_regime_avg = highlight_regime

            # Ensure metrics columns exist
            for col in ['Volatility', 'Sharpe Ratio', 'Max Drawdown']:
                if col not in trade_log_df.columns:
                    trade_log_df[col] = float('nan')
            for idx, row in trade_log_df.iterrows():
                asset = row['Asset']
                start_date = pd.to_datetime(row['Start Date'])
                end_date = pd.to_datetime(row['End Date'])
                asset_prices = merged_asset_data_metrics[['DateTime', asset]].dropna()
                mask = (asset_prices['DateTime'] >= start_date) & (asset_prices['DateTime'] <= end_date)
                price_series = asset_prices.loc[mask, asset]
                if len(price_series) > 1:
                    returns = price_series.pct_change().dropna()
                    vol = returns.std() * np.sqrt(12)
                    mean_ret = returns.mean() * 12
                    sharpe = mean_ret / (returns.std() * np.sqrt(12)) if returns.std() > 0 else float('nan')
                    cummax = price_series.cummax()
                    drawdown = (price_series - cummax) / cummax
                    max_dd = drawdown.min()
                    trade_log_df.at[idx, 'Volatility'] = vol
                    trade_log_df.at[idx, 'Sharpe Ratio'] = sharpe
                    trade_log_df.at[idx, 'Max Drawdown'] = max_dd
            # Reorder columns
            trade_log_df = trade_log_df[[
                'Regime', 'Start Date', 'End Date', 'Asset',
                'Start Price', 'End Price', 'Period Return',
                'Volatility', 'Sharpe Ratio', 'Max Drawdown'
            ]]
            # Format 'Start Date' and 'End Date' columns as YYYY-MM-DD (no time)
            trade_log_df['Start Date'] = trade_log_df['Start Date'].dt.strftime('%Y-%m-%d')
            trade_log_df['End Date'] = trade_log_df['End Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(
                trade_log_df.style
                    .format({
                        'Start Price': '{:.2f}',
                        'End Price': '{:.2f}',
                        'Period Return': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'Sharpe Ratio': '{:.2f}',
                        'Max Drawdown': '{:.2%}'
                    })
                    .apply(highlight_regime, axis=1),
                use_container_width=True
            )
            # --- FOOTNOTES for Trade Log Table ---
            st.markdown("""
**Column Definitions:**
- **Period Return**: (End Price - Start Price) / Start Price
- **Volatility**: Standard deviation of monthly returns within the period, annualized (multiplied by $\sqrt{12}$)
- **Sharpe Ratio**: Annualized mean monthly return divided by annualized volatility, assuming risk-free rate = 0
- **Max Drawdown**: Maximum observed loss from a peak to a trough during the period, based on monthly closing prices (as a percentage of the peak)

*Volatility and Sharpe ratio cannot be calculated for 1-month periods.*
""")

            # --- AGGREGATED AVERAGE RETURN TABLE (EXTENDED) ---
            required_cols = {'Period Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'}
            if required_cols.issubset(set(trade_log_df.columns)):
                avg_metrics = []
                # Add observed=False to groupby to silence FutureWarning and maintain current behavior
                for (regime, asset), group in trade_log_df.groupby(['Regime', 'Asset'], observed=False):
                    # Gather all monthly returns for this (regime, asset) group
                    # Find all periods for this group
                    monthly_returns = []
                    for idx, row in group.iterrows():
                        start_date = pd.to_datetime(row['Start Date'])
                        end_date = pd.to_datetime(row['End Date'])
                        # Get price series for this asset and date range
                        asset_prices = merged_asset_data_metrics[['DateTime', asset]].dropna()
                        mask = (asset_prices['DateTime'] >= start_date) & (asset_prices['DateTime'] <= end_date)
                        price_series = asset_prices.loc[mask, asset]
                        returns = price_series.pct_change().dropna()
                        monthly_returns.append(returns)
                    if monthly_returns:
                        all_returns = pd.concat(monthly_returns)
                        mean_monthly = all_returns.mean()
                        std_monthly = all_returns.std()
                        ann_return = mean_monthly * 12
                        ann_vol = std_monthly * (12 ** 0.5)
                        sharpe = ann_return / ann_vol if ann_vol > 0 else float('nan')
                    else:
                        ann_return = float('nan')
                        ann_vol = float('nan')
                        sharpe = float('nan')
                    # For drawdown, average period max drawdown (as originally calculated)
                    avg_drawdown = group['Max Drawdown'].mean()
                    avg_metrics.append({
                        'Regime': regime,
                        'Asset': asset,
                        'Annualized Return (Aggregated)': ann_return,
                        'Annualized Volatility (Aggregated)': ann_vol,
                        'Sharpe Ratio (Aggregated)': sharpe,
                        'Average Max Drawdown (Period Avg)': avg_drawdown,
                    })
                avg_metrics_table = pd.DataFrame(avg_metrics)
                regime_order = [
                    'Rising Growth & Falling Inflation',
                    'Rising Growth & Rising Inflation',
                    'Falling Growth & Falling Inflation',
                    'Falling Growth & Rising Inflation',
                ]
                avg_metrics_table['Regime'] = pd.Categorical(avg_metrics_table['Regime'], categories=regime_order, ordered=True)
                avg_metrics_table.sort_values(['Regime', 'Asset'], inplace=True)
                st.dataframe(
                    avg_metrics_table.style
                        .format({
                            'Annualized Return (Aggregated)': '{:.2%}',
                            'Annualized Volatility (Aggregated)': '{:.2%}',
                            'Sharpe Ratio (Aggregated)': '{:.2f}',
                            'Average Max Drawdown (Period Avg)': '{:.2%}'
                        })
                        .apply(highlight_regime_avg, axis=1),
                    use_container_width=True
                )
                # --- FOOTNOTES for Average Table --- (Revised for Clarity)
                st.markdown("""
**Aggregation & Calculation Notes:**
- **Annualized Return (Aggregated):** Calculated from the mean of *all* monthly returns across all periods within the regime/asset group, then annualized (multiplied by 12).
- **Annualized Volatility (Aggregated):** Calculated from the standard deviation of *all* monthly returns across all periods within the regime/asset group, then annualized (multiplied by √12).
- **Sharpe Ratio (Aggregated):** Calculated as `Annualized Return (Aggregated) / Annualized Volatility (Aggregated)`, assuming a 0% risk-free rate. Uses the aggregated metrics above.
- **Average Max Drawdown (Period Avg):** Calculated by averaging the 'Max Drawdown' values computed for *each individual period* within the regime/asset group.
- **Missing Data:** Periods or months with missing data (NaN) are excluded from calculations.
""")
                # --- BAR CHARTS for Average Metrics by Regime and Asset ---
                import plotly.graph_objects as go
                metrics_to_display = ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Sharpe Ratio (Aggregated)', 'Average Max Drawdown (Period Avg)']
                st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Bar Charts of Performance Metrics</h2>", unsafe_allow_html=True)
                for metric in metrics_to_display:
                    fig3 = go.Figure()
                    # Add regime background shading using shapes
                    unique_regimes = avg_metrics_table['Regime'].cat.categories
                    regime_x = list(unique_regimes)
                    for i, regime in enumerate(regime_x):
                        color = regime_bg_colors.get(
                            [k for k, v in regime_labels_dict.items() if v == regime][0],
                            'rgba(200,200,200,0.10)'
                        )
                        fig3.add_vrect(
                            x0=i - 0.5,
                            x1=i + 0.5,
                            fillcolor=color,
                            opacity=1.0,
                            layer="below",
                            line_width=0
                        )
                    for asset_name in avg_metrics_table['Asset'].unique():
                        asset_perf = avg_metrics_table[avg_metrics_table['Asset'] == asset_name]
                        fig3.add_trace(go.Bar(
                            x=asset_perf['Regime'],
                            y=asset_perf[metric],
                            name=asset_name,
                            marker_color=asset_colors.get(asset_name, 'gray')
                        ))
                    fig3.update_layout(
                        barmode='group',
                        xaxis_title='',
                        yaxis_title=metric,
                        title={
                            'text': metric + ' by Asset and Regime',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                        },
                        width=800,
                        height=500,
                        title_font=dict(size=20)
                    )
                    # Set y-axis formatting for percentage metrics
                    if metric in ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Average Max Drawdown (Period Avg)']:
                        fig3.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig3, use_container_width=False)
            else:
                st.warning("Missing required columns for average metrics table.")
        else:
            st.warning("No trade log data available.")
    else:
        st.warning("No trade log data available.")

# --- Ensure consistent title formatting on Tab 1 ---
# (Example for main chart, repeat as needed for all tab 1 charts)
fig.update_layout(
    title={
        'text': 'Macro Regime Timeline: S&P 500 & Inflation',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    },
    xaxis=dict(title='Date'),
    # yaxis title and other layout properties as before
)

# Repeat similar update_layout for all other tab 1 charts to match this style.
