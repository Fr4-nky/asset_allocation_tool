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
    page_icon="📈"
)

# Title and Description
st.title("Macroeconomic Regimes and Asset Performance Analysis")
st.write("""
This app visualizes macroeconomic regimes based on S&P 500 and Inflation Rate data, and analyzes asset performance across different regimes.
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
    inflation_url = "https://www.longtermtrends.net/data-inflation/"
    bonds_url = "https://www.longtermtrends.net/data-total-return-bond-index/"
    gold_url = "https://www.longtermtrends.net/data-gold-since-1792/"

    # Fetch each dataset (returns df with 'Date' as index or None)
    df_sp500 = fetch_and_decode(sp500_url, 'S&P 500')
    df_inflation = fetch_and_decode(inflation_url, 'Inflation Rate')
    df_bonds = fetch_and_decode(bonds_url, 'Bonds')
    df_gold = fetch_and_decode(gold_url, 'Gold')

    # --- ENSURE LEGACY METHODOLOGY: Monthly resampling (BME: business month end) ---
    for df in [df_sp500, df_inflation, df_bonds, df_gold]:
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            # Only resample if not already monthly
            if not (df.index.inferred_freq and df.index.inferred_freq.startswith('M')):
                df = df.resample('BME').last()

    # Merge asset classes on Date index
    asset_dfs = [df_sp500, df_bonds, df_gold]
    asset_ts_df = None
    try:
        asset_ts_df = pd.concat(asset_dfs, axis=1, join='outer')
        asset_ts_df = asset_ts_df.reset_index().rename(columns={'Date': 'DateTime'})
        asset_ts_df['DateTime'] = pd.to_datetime(asset_ts_df['DateTime'], errors='coerce')
        asset_ts_df = asset_ts_df.dropna(subset=['DateTime']).copy()
        print(f"SUCCESS: Asset class data loaded from APIs. Shape: {asset_ts_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load or merge asset data from APIs: {e}")
        asset_ts_df = pd.DataFrame()

    # --- Apply preprocessing steps from preprocessing3.ipynb ---
    print("DEBUG: Applying resampling and merging logic (like preprocessing3.ipynb)...")

    # 1. Resample to Business Month End ('BME')
    if df_sp500 is not None:
        print(f"DEBUG: Resampling S&P 500 (initial shape: {df_sp500.shape})")
        df_sp500 = df_sp500.resample('BME').last()
        print(f"DEBUG: Resampled S&P 500 shape: {df_sp500.shape}")
    if df_inflation is not None:
        print(f"DEBUG: Resampling Inflation Rate (initial shape: {df_inflation.shape})")
        df_inflation = df_inflation.resample('BME').last()
        print(f"DEBUG: Resampled Inflation Rate shape: {df_inflation.shape}")

    # 2. Inner Merge S&P 500 and Inflation Rate
    sp_inflation_df = pd.DataFrame() # Initialize empty df
    if df_sp500 is not None and df_inflation is not None:
        print("DEBUG: Performing INNER merge on resampled S&P 500 and Inflation Rate...")
        sp_inflation_df = pd.merge(df_sp500, df_inflation, left_index=True, right_index=True, how='inner')
        print(f"DEBUG: Inner merge result shape: {sp_inflation_df.shape}")
    elif df_sp500 is not None:
        print("WARN: Inflation data missing, using only S&P 500 data.")
        sp_inflation_df = df_sp500.copy()
        sp_inflation_df['Inflation Rate'] = np.nan # Add missing column
    elif df_inflation is not None:
        print("WARN: S&P 500 data missing, using only Inflation data.")
        sp_inflation_df = df_inflation.copy()
        sp_inflation_df['S&P 500'] = np.nan # Add missing column
    else:
        print("ERROR: Both S&P 500 and Inflation data failed to load or resample.")
        # sp_inflation_df remains empty

    # 3. Reset index to make 'Date' (or 'DateTime') a column
    if not sp_inflation_df.empty:
        sp_inflation_df = sp_inflation_df.reset_index()
        # Ensure the date column is named 'DateTime' as expected downstream
        if 'Date' in sp_inflation_df.columns and 'DateTime' not in sp_inflation_df.columns:
            sp_inflation_df = sp_inflation_df.rename(columns={'Date': 'DateTime'})
        elif 'index' in sp_inflation_df.columns and 'DateTime' not in sp_inflation_df.columns:
             sp_inflation_df = sp_inflation_df.rename(columns={'index': 'DateTime'})
        print("DEBUG: Reset index, 'DateTime' column created.")

    # --- Apply >= 2000 Filter AFTER resampling/merging ---
    filter_date = pd.Timestamp('1972-01-01')
    print(f"DEBUG: Applying final filter for dates >= {filter_date}...")

    if not sp_inflation_df.empty:
        if 'DateTime' in sp_inflation_df.columns:
            sp_inflation_df['DateTime'] = pd.to_datetime(sp_inflation_df['DateTime']) # Ensure datetime type
            original_rows_before_2000_filter = len(sp_inflation_df)
            sp_inflation_df = sp_inflation_df[sp_inflation_df['DateTime'] >= filter_date].copy()
            print(f"DEBUG: Filtered resampled/merged S&P/Inflation data >= {filter_date}. Shape: {sp_inflation_df.shape}. (Was {original_rows_before_2000_filter} rows before)")
        else:
             print("ERROR: 'DateTime' column missing before final 2000 filter for S&P/Inflation data.")
             sp_inflation_df = pd.DataFrame() # Make empty if date column is lost

    if not asset_ts_df.empty:
        if 'DateTime' in asset_ts_df.columns:
            asset_ts_df['DateTime'] = pd.to_datetime(asset_ts_df['DateTime']) # Ensure datetime type
            asset_ts_df = asset_ts_df[asset_ts_df['DateTime'] >= filter_date].copy()
            print(f"DEBUG: Filtered Asset data >= {filter_date}. Shape: {asset_ts_df.shape}")
        else:
            print("ERROR: 'DateTime' column missing before final 2000 filter for Asset data.")
            asset_ts_df = pd.DataFrame() # Make empty

    print("LOG: Loaded sp_inflation_data: shape={}, min_date={}, max_date={}".format(sp_inflation_df.shape, sp_inflation_df['DateTime'].min(), sp_inflation_df['DateTime'].max()))
    print("LOG: Loaded asset_ts_data: shape={}, min_date={}, max_date={}".format(asset_ts_df.shape, asset_ts_df['DateTime'].min(), asset_ts_df['DateTime'].max()))
    print("DEBUG: load_data function finished.")
    # Return the processed S&P/Inflation df and the filtered asset df
    return sp_inflation_df.copy(), asset_ts_df.copy()

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
st.sidebar.header("User Input Parameters")

# Tabs for S&P 500 and Inflation Rate Parameters
param_tabs = st.sidebar.tabs(["S&P 500 Parameters", "Inflation Rate Parameters"])

# S&P 500 Parameters
with param_tabs[0]:
    st.subheader("S&P 500 Parameters")
    # Rolling Window Size Input
    sp500_n = st.number_input(
        "S&P 500 Moving Average Window (months):",
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        key='sp500_n',
        help="Number of months for the moving average window."
    )

# Inflation Rate Parameters
with param_tabs[1]:
    st.subheader("Inflation Rate Parameters")
    # Rolling Window Size Input
    inflation_n = st.number_input(
        "Inflation Rate Moving Average Window (months):",
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        key='inflation_n',
        help="Number of months for the moving average window."
    )

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
    st.subheader("Regime Visualization")
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
            # Set end date to one day before the next regime's start date
            end_date_regime = regime_periods_df.loc[i+1, 'Start Date'] - pd.Timedelta(days=1)
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

        # Select data, fill NaNs that might interfere with stacking (e.g., with 0 or a placeholder)
        # Choose a fill value appropriate for your data, or handle NaNs differently if needed
        sp500_data = sp_inflation_data['S&P 500'].fillna(0)
        sp500_ma_data = sp_inflation_data['S&P 500 MA'].fillna(0)
        inflation_data = sp_inflation_data['Inflation Rate'].fillna(0)
        inflation_ma_data = sp_inflation_data['Inflation Rate MA'].fillna(0)

        customdata = np.stack((
            regime_labels,
            sp500_data,
            sp500_ma_data,
            inflation_data,
            inflation_ma_data
        ), axis=-1)
        print(f"DEBUG: Tab 1 - Customdata array created successfully. Shape: {customdata.shape}") # Added debug print
    except Exception as e:
        print(f"ERROR: Tab 1 - Failed to create customdata array: {e}")
        st.error(f"Failed to prepare data for plotting: {e}")
        # Assign a dummy array or stop execution if customdata is critical
        customdata = np.empty((len(sp_inflation_data), 5)) # Example dummy
        # Or potentially st.stop() here if the plot can't proceed

    # Add traces based on user selection, reusing the customdata array
    print("DEBUG: Tab 1 - Starting add_trace section.") # Added debug print
    hover_regime = 'Regime: %{customdata[0]}<extra></extra>'
    if show_sp500_ma:
        print("DEBUG: Tab 1 - Preparing S&P 500 MA trace...")
        # REUSE customdata
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500 MA'], # Use original MA data for plotting Y
            mode='lines',
            name=f'S&P 500 MA ({sp500_n}m)',
            line=dict(color=asset_colors['S&P 500 MA']),
            yaxis='y1',
            customdata=customdata, # Use the pre-calculated customdata
            hovertemplate='S&P 500 MA: %{customdata[2]:.2f}<br>' + hover_regime
        ))
        print("DEBUG: Tab 1 - Added S&P 500 MA trace.")
    if show_inflation_ma:
        print("DEBUG: Tab 1 - Preparing Inflation Rate MA trace...")
        # REUSE customdata
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate MA'], # Use original MA data for plotting Y
            mode='lines',
            name=f'Inflation Rate MA ({inflation_n}m)',
            line=dict(color=asset_colors['Inflation Rate MA']),
            yaxis='y2',
            customdata=customdata, # Use the pre-calculated customdata
            hovertemplate='Inflation Rate MA: %{customdata[4]:.2f}<br>' + hover_regime
        ))
        print("DEBUG: Tab 1 - Added Inflation Rate MA trace.")
    if show_sp500:
        print("DEBUG: Tab 1 - Preparing S&P 500 trace...")
        # REUSE customdata
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500'], # Use original S&P data for plotting Y
            mode='lines',
            name='S&P 500',
            line=dict(color=asset_colors['S&P 500'], dash='dot'),
            yaxis='y1',
            customdata=customdata, # Use the pre-calculated customdata
            hovertemplate='S&P 500: %{customdata[1]:.2f}<br>' + hover_regime
        ))
        print("DEBUG: Tab 1 - Added S&P 500 trace.")
    if show_inflation:
        print("DEBUG: Tab 1 - Preparing Inflation Rate trace...")
        # REUSE customdata
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate'], # Use original Inflation data for plotting Y
            mode='lines',
            name='Inflation Rate',
            line=dict(color=asset_colors['Inflation Rate'], dash='dot'),
            yaxis='y2',
            customdata=customdata, # Use the pre-calculated customdata
            hovertemplate='Inflation Rate: %{customdata[3]:.2f}<br>' + hover_regime
        ))
        print("DEBUG: Tab 1 - Added Inflation Rate trace.")
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
    st.markdown("### Regime Legend with Definitions")
    regime_legend_html = "<ul style='list-style-type:none;'>"
    
    # Add numeric regimes
    for regime_num in sorted(regime_labels_dict.keys(), key=lambda x: int(x) if x != 'Unknown' else float('inf')):
        if regime_num == 'Unknown':
            continue
        color = regime_colors.get(regime_num, 'grey')
        label = regime_labels_dict.get(regime_num, 'Unknown')
        regime_legend_html += f"<li><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px;'></span> <b>{label}</b></li>"
    regime_legend_html += "</ul>"
    st.markdown(regime_legend_html, unsafe_allow_html=True)
    
    # Export plot as image
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png')
    st.download_button(
        label="Download Plot as PNG",
        data=buffer,
        file_name='regime_plot.png',
        mime='image/png'
    )
    
    # Provide a download button for the regime data
    regime_download_df = sp_inflation_data[['DateTime', 'Regime']].copy()
    regime_download_df['Regime Label'] = regime_download_df['Regime'].map(regime_labels_dict)
    csv = regime_download_df.to_csv(index=False)
    st.download_button(
        label="Download Regime Data as CSV",
        data=csv,
        file_name='regime_data.csv',
        mime='text/csv',
    )
    
    # Add Regime Diagrams under the legends
    st.markdown("## Regime Diagrams")
    
    ### Diagram 1: 2D Scatter Plot of Growth Rates with Regime Boundaries
    st.markdown("### 1. 2D Scatter Plot of Growth Rates with Regime Boundaries")
    print("DEBUG: Tab 1 - Preparing Scatter Plot...") # Added debug print
    # Prepare data for plotting
    derivative_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna()
    derivative_df['Regime Label'] = derivative_df['Regime'].map(regime_labels_dict)
    
    # Sort by DateTime
    derivative_df = derivative_df.sort_values('DateTime').reset_index(drop=True)
    n_points = len(derivative_df)
    window_size = 50
    # Show only the latest 50 data points for scatterplot
    window_df = derivative_df.iloc[-window_size:]
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
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500 MA Growth: %{x:.4f}<br>' +
            'Inflation Rate MA Growth: %{y:.4f}<extra></extra>'
        ),
        name=date_span
    ))
    scatter_fig.update_layout(
        xaxis_title='S&P 500 MA Growth',
        yaxis_title='Inflation Rate MA Growth',
        title={
            'text': f'Market & Inflation Momentum ({date_span})',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        width=800,
        height=600,
        showlegend=False
    )
    st.plotly_chart(scatter_fig)
    print("DEBUG: Tab 1 Scatter Plot complete.") # Added debug print
    
    ### Diagram 2: Interactive Scatter Plot of Growth Values with Regime Boundaries (All Data, Color Gradient)
    st.markdown("### 2. Interactive Scatter Plot of Growth Values with Regime Boundaries (All Data, Color Gradient)")
    print("DEBUG: Tab 1 - Preparing All-Data Scatter Plot with Color Gradient...")
    all_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna().sort_values('DateTime')
    N_all = len(all_df)
    by_cmap_all = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=N_all)
    all_colors = [mcolors.to_hex(by_cmap_all(i/(N_all-1))) for i in range(N_all)]
    all_scatter_fig = go.Figure()
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
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500 MA Growth: %{x:.4f}<br>' +
            'Inflation Rate MA Growth: %{y:.4f}<extra></extra>'
        ),
        name='All Data (Oldest to Newest)'
    ))
    all_scatter_fig.update_layout(
        xaxis_title='S&P 500 MA Growth',
        yaxis_title='Inflation Rate MA Growth',
        title={
            'text': 'Scatter Plot of Growth with Regime Boundaries (All Data, Color Gradient)',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        width=800,
        height=600,
        showlegend=False
    )
    st.plotly_chart(all_scatter_fig)
    print("DEBUG: Tab 1 All-Data Scatter Plot complete.")

t_tab1_end = time.time()
print(f"DEBUG: Tab 1 rendering took {t_tab1_end-t_tab1:.2f} seconds.")

# Tab 2: Asset Performance & Metrics
with tabs[1]:
    st.subheader("Asset Performance Over Time")
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
            end_date_regime = regime_periods_df.loc[i+1, 'Start Date'] - pd.Timedelta(days=1)
        else:
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
    fig2.update_layout(shapes=shapes)
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
        asset_data['Normalized Price'] = asset_data['Actual Price'] / asset_data['Actual Price'].iloc[0] * 100
        customdata = np.stack((asset_data['Actual Price'], asset_data['Regime Label']), axis=-1)
        price_label = 'Actual Price'
        hovertemplate=(
            asset + "<br>"
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

    # Update layout
    t_layout_start = time.time()
    fig2.update_layout(
        title='Asset Performance Over Time (Normalized to 100 at First Available Date)',
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
            font_family="Arial"
        )
    )
    t_layout_end = time.time()
    print(f"DEBUG: Tab 2 - Layout update took {t_layout_end-t_layout_start:.3f} seconds.")

    st.plotly_chart(fig2, use_container_width=False)

    # Download buttons for plot and data
    buffer = io.BytesIO()
    fig2.write_image(buffer, format='png')
    st.download_button(
        label="Download Asset Performance Plot as PNG",
        data=buffer,
        file_name='asset_performance_plot.png',
        mime='image/png',
    )
    all_asset_data = merged_asset_data[['DateTime'] + ['S&P 500', 'Gold', 'Bonds'] + ['Regime']].copy()
    csv = all_asset_data.to_csv(index=False)
    st.download_button(
        label="Download Asset Data as CSV",
        data=csv,
        file_name='asset_data.csv',
        mime='text/csv',
    )

    # --- Performance Metrics per Regime (from former Tab 3) ---
    st.subheader("Performance Metrics per Regime")
    print("DEBUG: Rendering Performance Metrics Section (from former Tab 3).")
    t_metrics_start = time.time()
    with st.spinner('Merging asset data with regimes...'):
        merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
        print(f"DEBUG: Tab 2 - Asset data merged (for metrics). Shape: {merged_asset_data_metrics.shape}")
    performance_results = []
    t_metrics_loop_start = time.time()
    for asset in ['S&P 500', 'Gold', 'Bonds']:
        asset_data = merged_asset_data_metrics[['DateTime', asset, 'Regime']].copy()
        asset_data = asset_data[asset_data['Regime'] != 'Unknown'].copy()
        asset_data = asset_data.dropna(subset=[asset, 'Regime']).copy()
        asset_data['Regime Label'] = asset_data['Regime'].map(regime_labels_dict)
        asset_data['Return'] = asset_data[asset].pct_change()
        asset_data = asset_data.dropna(subset=['Return']).copy()
        for regime in asset_data['Regime'].unique():
            regime_data = asset_data[asset_data['Regime'] == regime].copy()
            t_metrics_inner_start = time.time()
            if len(regime_data) < 2:
                performance_results.append({
                    'Asset': asset,
                    'Regime': regime_labels_dict.get(regime, 'Unknown'),
                    'Average Return': np.nan,
                    'Volatility': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Max Drawdown': np.nan
                })
                t_metrics_inner_end = time.time()
                print(f"DEBUG: Tab 2 - Asset {asset} regime {regime} (SKIP) metrics calc took {t_metrics_inner_end-t_metrics_inner_start:.3f} seconds.")
                continue
            avg_return = regime_data['Return'].mean() * 252
            volatility = regime_data['Return'].std() * np.sqrt(252)
            sharpe_ratio = avg_return / volatility if volatility != 0 else np.nan
            cumulative = (1 + regime_data['Return']).cumprod()
            cumulative_max = cumulative.cummax()
            drawdown = cumulative / cumulative_max - 1
            max_drawdown = drawdown.min()
            performance_results.append({
                'Asset': asset,
                'Regime': regime_labels_dict.get(regime, 'Unknown'),
                'Average Return': avg_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown
            })
            t_metrics_inner_end = time.time()
            print(f"DEBUG: Tab 2 - Asset {asset} regime {regime} metrics calc took {t_metrics_inner_end-t_metrics_inner_start:.3f} seconds.")
    t_metrics_loop_end = time.time()
    print(f"DEBUG: Tab 2 - Performance metrics asset/loop took {t_metrics_loop_end-t_metrics_loop_start:.3f} seconds.")
    t_metrics_end = time.time()
    print(f"DEBUG: Tab 2 - Performance metrics computation took {t_metrics_end-t_metrics_start:.3f} seconds.")

    perf_data_filtered = pd.DataFrame(performance_results)
    if perf_data_filtered.empty:
        st.warning("No performance data available for the selected options.")
    else:
        # Bar Charts for each metric (with regime background coloring)
        metrics_to_display = ['Average Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
        for metric in metrics_to_display:
            st.markdown(f"#### {metric} by Asset and Regime")
            fig3 = go.Figure()
            # Add regime background shading using shapes
            unique_regimes = perf_data_filtered['Regime'].unique()
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
            for asset_name in perf_data_filtered['Asset'].unique():
                asset_perf = perf_data_filtered[perf_data_filtered['Asset'] == asset_name]
                fig3.add_trace(go.Bar(
                    x=asset_perf['Regime'],
                    y=asset_perf[metric],
                    name=asset_name,
                    marker_color=asset_colors.get(asset_name, 'gray')
                ))
            fig3.update_layout(
                barmode='group',
                xaxis_title='Regime',
                yaxis_title=metric,
                title=f'{metric} by Asset and Regime',
                width=800,
                height=500
            )
            st.plotly_chart(fig3, use_container_width=False)
        # Show the table at the very bottom
        st.dataframe(perf_data_filtered)
        csv = perf_data_filtered.to_csv(index=False)
        st.download_button(
            label="Download Performance Metrics Data as CSV",
            data=csv,
            file_name='performance_metrics.csv',
            mime='text/csv',
        )
    t_tab2_end = time.time()
    print(f"DEBUG: Tab 2 rendering took {t_tab2_end-t_tab2:.2f} seconds.")

    print("DEBUG: End of script execution.") # Added debug print
