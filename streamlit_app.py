# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import re
import concurrent.futures

from data.fetch import fetch_and_decode, decode_base64_data
from data.processing import merge_asset_with_regimes, compute_moving_average, compute_growth, assign_regimes
from viz.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from metrics.performance import generate_aggregated_metrics
from config.constants import asset_colors, regime_bg_colors, regime_legend_colors, regime_labels_dict, asset_list_tab2, asset_list_tab3, asset_list_tab4, asset_list_tab5, asset_list_tab6, regime_definitions, REGIME_BG_ALPHA

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

# Load Data Function
@st.cache_data
def load_data():
    # --- Define all data URLs ---
    sp500_url = "https://www.longtermtrends.net/data-sp500-since-1871/"
    inflation_url = "https://www.longtermtrends.net/data-inflation-forecast/"
    bonds_url = "https://www.longtermtrends.net/data-total-return-bond-index/"
    gold_url = "https://www.longtermtrends.net/data-gold-since-1792/"
    # MSCI USA series
    msci_large_url = "https://www.longtermtrends.net/data-msci-usa-large-cap/"
    msci_mid_url   = "https://www.longtermtrends.net/data-msci-usa-mid-cap/"
    msci_small_url = "https://www.longtermtrends.net/data-msci-usa-small-cap/"
    msci_micro_url = "https://www.longtermtrends.net/data-msci-usa-micro-cap/"
    msci_cyclical_url = "https://www.longtermtrends.net/data-msci-cyclical-stocks/"
    msci_defensive_url = "https://www.longtermtrends.net/data-msci-defensive-stocks/"
    # US Sector URLs
    comm_url = "https://www.longtermtrends.net/data-us-communication-services/"
    mat_url = "https://www.longtermtrends.net/data-us-basic-materials/"
    energy_url = "https://www.longtermtrends.net/data-us-energy/"
    financial_url = "https://www.longtermtrends.net/data-us-financial/"
    industrial_url = "https://www.longtermtrends.net/data-us-industrial/"
    technology_url = "https://www.longtermtrends.net/data-us-technology/"
    cons_stap_url = "https://www.longtermtrends.net/data-us-consumer-staples/"
    utilities_url = "https://www.longtermtrends.net/data-us-utiliiies/"
    health_url = "https://www.longtermtrends.net/data-us-thcare/"
    cons_disc_url = "https://www.longtermtrends.net/data-us-consumer-discretionary/"
    real_estate_url = "https://www.longtermtrends.net/data-us-real-estate/"
    # MSCI World Factor Strategy URLs
    world_mom_url = "https://www.longtermtrends.net/data-msci-world-momentum/"
    world_growth_url = "https://www.longtermtrends.net/data-msci-world-growth-target/"
    world_quality_url = "https://www.longtermtrends.net/data-msci-world-quality/"
    world_gov_url = "https://www.longtermtrends.net/data-msci-world-governance-quality/"
    world_div_mast_url = "https://www.longtermtrends.net/data-msci-world-dividend-masters/"
    world_high_div_url = "https://www.longtermtrends.net/data-msci-world-high-dividend-yield/"
    world_buyback_url = "https://www.longtermtrends.net/data-msci-world-buy-back-yield/"
    world_tsy_url = "https://www.longtermtrends.net/data-msci-world-total-shareholder-yield/"
    world_small_url = "https://www.longtermtrends.net/data-msci-world-small-cap/"
    world_ew_url = "https://www.longtermtrends.net/data-msci-world-equal-weighted/"
    world_enh_val_url = "https://www.longtermtrends.net/data-msci-world-enhanced-value/"
    world_prime_val_url = "https://www.longtermtrends.net/data-msci-world-prime-value/"
    world_min_vol_url = "https://www.longtermtrends.net/data-msci-minimum-volatility/"
    world_risk_url = "https://www.longtermtrends.net/data-msci-world-risk-weighted/"

    # --- Fetch all datasets in parallel ---
    urls = {
        'S&P 500': sp500_url,
        'Inflation Rate': inflation_url,
        'Bonds': bonds_url,
        'Gold': gold_url,
        'MSCI USA Large Cap': msci_large_url,
        'MSCI USA Mid Cap': msci_mid_url,
        'MSCI USA Small Cap': msci_small_url,
        'MSCI USA Micro Cap': msci_micro_url,
        'MSCI USA Cyclical Stocks': msci_cyclical_url,
        'MSCI USA Defensive Stocks': msci_defensive_url,
        'US Communication Services': comm_url,
        'US Basic Materials': mat_url,
        'US Energy': energy_url,
        'US Financial': financial_url,
        'US Industrial': industrial_url,
        'US Technology': technology_url,
        'US Consumer Staples': cons_stap_url,
        'US Utilities': utilities_url,
        'US Health Care': health_url,
        'US Consumer Discretionary': cons_disc_url,
        'US Real Estate': real_estate_url,
        # Factor strategies
        'MSCI World Momentum': world_mom_url,
        'MSCI World Growth Target': world_growth_url,
        'MSCI World Quality': world_quality_url,
        'MSCI World Governance Quality': world_gov_url,
        'MSCI World Dividend Masters': world_div_mast_url,
        'MSCI World High Dividend Yield': world_high_div_url,
        'MSCI World Buyback Yield': world_buyback_url,
        'MSCI World Total Shareholder Yield': world_tsy_url,
        'MSCI World Small Cap': world_small_url,
        'MSCI World Equal Weighted': world_ew_url,
        'MSCI World Enhanced Value': world_enh_val_url,
        'MSCI World Prime Value': world_prime_val_url,
        'MSCI World Minimum Volatility (USD)': world_min_vol_url,
        'MSCI World Risk Weighted': world_risk_url
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_name = {executor.submit(fetch_and_decode, url, name): name for name, url in urls.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"ERROR fetching {name}: {e}")
                results[name] = None

    # Unpack results
    df_sp500 = results.get('S&P 500')
    df_inflation = results.get('Inflation Rate')
    df_bonds = results.get('Bonds')
    df_gold = results.get('Gold')
    df_msci_large = results.get('MSCI USA Large Cap')
    df_msci_mid = results.get('MSCI USA Mid Cap')
    df_msci_small = results.get('MSCI USA Small Cap')
    df_msci_micro = results.get('MSCI USA Micro Cap')
    df_msci_cyclical = results.get('MSCI USA Cyclical Stocks')
    df_msci_defensive = results.get('MSCI USA Defensive Stocks')
    df_comm = results.get('US Communication Services')
    df_mat = results.get('US Basic Materials')
    df_energy = results.get('US Energy')
    df_financial = results.get('US Financial')
    df_industrial = results.get('US Industrial')
    df_technology = results.get('US Technology')
    df_cons_stap = results.get('US Consumer Staples')
    df_utilities = results.get('US Utilities')
    df_health = results.get('US Health Care')
    df_cons_disc = results.get('US Consumer Discretionary')
    df_real_estate = results.get('US Real Estate')
    # Factor data
    df_world_momentum = results.get('MSCI World Momentum')
    df_world_growth = results.get('MSCI World Growth Target')
    df_world_quality = results.get('MSCI World Quality')
    df_world_gov = results.get('MSCI World Governance Quality')
    df_world_div_mast = results.get('MSCI World Dividend Masters')
    df_world_high_div = results.get('MSCI World High Dividend Yield')
    df_world_buyback = results.get('MSCI World Buyback Yield')
    df_world_tsy = results.get('MSCI World Total Shareholder Yield')
    df_world_small = results.get('MSCI World Small Cap')
    df_world_ew = results.get('MSCI World Equal Weighted')
    df_world_enh_val = results.get('MSCI World Enhanced Value')
    df_world_prime_val = results.get('MSCI World Prime Value')
    df_world_min_vol = results.get('MSCI World Minimum Volatility (USD)')
    df_world_risk = results.get('MSCI World Risk Weighted')

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
    df_msci_large_resampled = resample_and_correct_date(df_msci_large, 'MSCI USA Large Cap')
    df_msci_mid_resampled = resample_and_correct_date(df_msci_mid, 'MSCI USA Mid Cap')
    df_msci_small_resampled = resample_and_correct_date(df_msci_small, 'MSCI USA Small Cap')
    df_msci_micro_resampled = resample_and_correct_date(df_msci_micro, 'MSCI USA Micro Cap')
    df_msci_cyclical_resampled = resample_and_correct_date(df_msci_cyclical, 'MSCI USA Cyclical Stocks')
    df_msci_defensive_resampled = resample_and_correct_date(df_msci_defensive, 'MSCI USA Defensive Stocks')
    # Resample US Sector series
    df_comm_resampled = resample_and_correct_date(df_comm, 'US Communication Services')
    df_mat_resampled = resample_and_correct_date(df_mat, 'US Basic Materials')
    df_energy_resampled = resample_and_correct_date(df_energy, 'US Energy')
    df_financial_resampled = resample_and_correct_date(df_financial, 'US Financial')
    df_industrial_resampled = resample_and_correct_date(df_industrial, 'US Industrial')
    df_technology_resampled = resample_and_correct_date(df_technology, 'US Technology')
    df_cons_stap_resampled = resample_and_correct_date(df_cons_stap, 'US Consumer Staples')
    df_utilities_resampled = resample_and_correct_date(df_utilities, 'US Utilities')
    df_health_resampled = resample_and_correct_date(df_health, 'US Health Care')
    df_cons_disc_resampled = resample_and_correct_date(df_cons_disc, 'US Consumer Discretionary')
    df_real_estate_resampled = resample_and_correct_date(df_real_estate, 'US Real Estate')
    # Resample factor series
    df_world_momentum_resampled = resample_and_correct_date(df_world_momentum, 'MSCI World Momentum')
    df_world_growth_resampled = resample_and_correct_date(df_world_growth, 'MSCI World Growth Target')
    df_world_quality_resampled = resample_and_correct_date(df_world_quality, 'MSCI World Quality')
    df_world_gov_resampled = resample_and_correct_date(df_world_gov, 'MSCI World Governance Quality')
    df_world_div_mast_resampled = resample_and_correct_date(df_world_div_mast, 'MSCI World Dividend Masters')
    df_world_high_div_resampled = resample_and_correct_date(df_world_high_div, 'MSCI World High Dividend Yield')
    df_world_buyback_resampled = resample_and_correct_date(df_world_buyback, 'MSCI World Buyback Yield')
    df_world_tsy_resampled = resample_and_correct_date(df_world_tsy, 'MSCI World Total Shareholder Yield')
    df_world_small_resampled = resample_and_correct_date(df_world_small, 'MSCI World Small Cap')
    df_world_ew_resampled = resample_and_correct_date(df_world_ew, 'MSCI World Equal Weighted')
    df_world_enh_val_resampled = resample_and_correct_date(df_world_enh_val, 'MSCI World Enhanced Value')
    df_world_prime_val_resampled = resample_and_correct_date(df_world_prime_val, 'MSCI World Prime Value')
    df_world_min_vol_resampled = resample_and_correct_date(df_world_min_vol, 'MSCI World Minimum Volatility (USD)')
    df_world_risk_resampled = resample_and_correct_date(df_world_risk, 'MSCI World Risk Weighted')

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
    all_asset_dfs = [df for df in [df_sp500_resampled, df_gold_resampled, df_bonds_resampled,
                                   df_msci_large_resampled, df_msci_mid_resampled,
                                   df_msci_small_resampled, df_msci_micro_resampled,
                                   df_msci_cyclical_resampled, df_msci_defensive_resampled,
                                   df_comm_resampled, df_mat_resampled, df_energy_resampled,
                                   df_financial_resampled, df_industrial_resampled,
                                   df_technology_resampled, df_cons_stap_resampled,
                                   df_utilities_resampled, df_health_resampled,
                                   df_cons_disc_resampled, df_real_estate_resampled,
                                   df_world_momentum_resampled, df_world_growth_resampled,
                                   df_world_quality_resampled, df_world_gov_resampled,
                                   df_world_div_mast_resampled, df_world_high_div_resampled,
                                   df_world_buyback_resampled, df_world_tsy_resampled,
                                   df_world_small_resampled, df_world_ew_resampled,
                                   df_world_enh_val_resampled, df_world_prime_val_resampled,
                                   df_world_min_vol_resampled, df_world_risk_resampled] if df is not None]
    if len(all_asset_dfs) > 1:
        print(f"DEBUG: Performing OUTER merge on {len(all_asset_dfs)} resampled asset DataFrames...")
        # Ensure all dataframes have the same index name before merging
        base_index_name = all_asset_dfs[0].index.name if all_asset_dfs[0].index.name else 'Date'
        for i, df in enumerate(all_asset_dfs):
            if df.index.name != base_index_name:
                print(f"WARN: Aligning index name for asset df {i} ('{df.index.name}' -> '{base_index_name}')")
                # Decide on a common name, e.g., 'Date' or use the first df's name
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

# --- User-defined Start and End Dates BEFORE computing MAs and Growth ---
# Determine full data span
min_date = min(sp_inflation_data['DateTime'].min(), asset_ts_data['DateTime'].min()).date()
max_date = max(sp_inflation_data['DateTime'].max(), asset_ts_data['DateTime'].max()).date()
# Separate sidebar inputs
start_date = st.sidebar.date_input(
    "Start Date", value=min_date, min_value=min_date, max_value=max_date, key="start_date"
)
end_date = st.sidebar.date_input(
    "End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date"
)
if start_date > end_date:
    st.sidebar.error("Start Date must be on or before End Date")
    st.stop()
# Debug logging
print(f"DEBUG: Selected date range: {start_date} to {end_date}")
# Convert to Timestamps
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
# Filter both datasets
sp_inflation_data = sp_inflation_data[(sp_inflation_data['DateTime'] >= start_date) & (sp_inflation_data['DateTime'] <= end_date)].copy()
asset_ts_data = asset_ts_data[(asset_ts_data['DateTime'] >= start_date) & (asset_ts_data['DateTime'] <= end_date)].copy()
print(f"DEBUG: After filtering: SP shape {sp_inflation_data.shape}, Asset shape {asset_ts_data.shape}")

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
    # --- DEBUG PRINTS FOR GROWTH COLUMNS ---
    print("DEBUG: S&P 500 MA Growth min/max:", sp_inflation_data['S&P 500 MA Growth'].min(), sp_inflation_data['S&P 500 MA Growth'].max())
    print("DEBUG: Inflation Rate MA Growth min/max:", sp_inflation_data['Inflation Rate MA Growth'].min(), sp_inflation_data['Inflation Rate MA Growth'].max())
    print("DEBUG: S&P 500 MA Growth sample:", sp_inflation_data['S&P 500 MA Growth'].head())
    print("DEBUG: Inflation Rate MA Growth sample:", sp_inflation_data['Inflation Rate MA Growth'].head())
    # --- DO NOT DROP ROWS WITH NANs IN MA OR GROWTH COLUMNS BEFORE REGIME ASSIGNMENT OR PLOTTING (MATCH OLD BEHAVIOR) ---
    # sp_inflation_data = sp_inflation_data.dropna(subset=[
    #     'S&P 500 MA', 'Inflation Rate MA', 'S&P 500 MA Growth', 'Inflation Rate MA Growth'
    # ]).copy()
    # print("DEBUG: After dropna, sp_inflation_data shape:", sp_inflation_data.shape)
t1 = time.time()
print(f"DEBUG: Growth computation took {t1-t0:.2f} seconds.")

# Now that we have the growth, we can get min and max values
sp500_growth = sp_inflation_data['S&P 500 MA Growth'].dropna()
inflation_growth = sp_inflation_data['Inflation Rate MA Growth'].dropna()

sp500_min = float(sp500_growth.min())
sp500_max = float(sp500_growth.max())
inflation_min = float(inflation_growth.min())
inflation_max = float(inflation_growth.max())

# --- Logging for Regime Assignment ---
t0 = time.time()
with st.spinner('Assigning Regimes...'):
    sp_inflation_data = assign_regimes(sp_inflation_data, regime_definitions)
    print("DEBUG: Regimes assigned.")
t1 = time.time()
print(f"DEBUG: Regime assignment took {t1-t0:.2f} seconds.")

# Handle any NaN regimes (should not happen)
sp_inflation_data['Regime'] = sp_inflation_data['Regime'].fillna('Unknown')

# --- Logging for Tab Rendering ---
t0 = time.time()
print("DEBUG: Starting Tab rendering.")
tabs = st.tabs(["Regime Visualization", "Asset Classes", "Large vs. Small Cap", "Cyclical vs. Defensive", "US Sectors", "Factor Investing"])
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
    regime_legend_html = "<ul style='list-style-type:none; padding-left:0;'>"
    # Custom legend names: Goldilocks, Reflation, Stagflation, Deflation
    custom_legend_names = {
        1: "<b>Reflation</b>: Rising growth, rising inflation",
        2: "<b>Goldilocks</b>: Rising growth, falling inflation",
        3: "<b>Stagflation</b>: Falling growth, rising inflation",
        4: "<b>Deflation</b>: Falling growth, falling inflation"
    }
    # custom order: 2,1,4,3
    for regime_num in [2,1,4,3]:
         color = regime_bg_colors.get(regime_num, 'grey')
         label = custom_legend_names.get(regime_num, regime_labels_dict.get(regime_num, 'Unknown'))
         regime_legend_html += f"<li><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px; border-radius:3px; border:1px solid #888;'></span> {label}</li>"
    regime_legend_html += "</ul>"
    st.markdown(regime_legend_html, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Scatter Plots</h2>", unsafe_allow_html=True)
    
    # --- First Scatter Plot ---
    # Prepare data for plotting
    derivative_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna()
    # Add S&P 500 MA pct change for scatter x-axis
    derivative_df['S&P 500 MA Pct Change'] = sp_inflation_data['S&P 500 MA'].pct_change()
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
    x_min = float(window_df['S&P 500 MA Pct Change'].min()) if not window_df.empty else -1
    x_max = float(window_df['S&P 500 MA Pct Change'].max()) if not window_df.empty else 1
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
    # Update scatter plot to use S&P 500 MA pct change on x-axis
    scatter_fig.add_trace(go.Scatter(
        x=window_df['S&P 500 MA Pct Change'],
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
            window_df['S&P 500 MA Pct Change'],
            window_df['Inflation Rate MA Growth'],
            window_df['Regime'] if 'Regime' in window_df else np.full(len(window_df), ''),
        ], axis=-1),
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500 MA % Change: %{customdata[0]:.2%}<br>' +
            'Inflation Rate MA Growth: %{customdata[1]:.2%}<br>' +
            'Regime: %{customdata[2]}<extra></extra>'
        ),
        name=date_span
    ))
    scatter_fig.update_layout(
        xaxis_title='S&P 500 MA % Change',
        yaxis_title='Inflation Rate MA Growth',
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
    all_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime', 'S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']].dropna()
    # Add S&P 500 MA pct change for scatter x-axis
    all_df['S&P 500 MA Pct Change'] = sp_inflation_data['S&P 500 MA'].pct_change()
    for col in ['S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']:
        if col not in all_df.columns:
            all_df.loc[:, col] = np.nan
    N_all = len(all_df)
    by_cmap_all = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=N_all)
    all_colors = [mcolors.to_hex(by_cmap_all(i/(N_all-1))) for i in range(N_all)]
    all_scatter_fig = go.Figure()
    # Determine axis ranges for quadrants (for all_scatter_fig)
    x_min_all = float(all_df['S&P 500 MA Pct Change'].min()) if not all_df.empty else -1
    x_max_all = float(all_df['S&P 500 MA Pct Change'].max()) if not all_df.empty else 1
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
        x=all_df['S&P 500 MA Pct Change'],
        y=all_df['Inflation Rate MA Growth'],
        mode='markers',
        marker=dict(
            color=all_colors,
            size=10,
            line=dict(width=1, color='black')
        ),
        text=all_df['DateTime'].dt.strftime('%Y-%m-%d'),
        customdata=np.stack([
            all_df['S&P 500 MA Pct Change'],
            all_df['Inflation Rate MA Growth'],
            all_df['Regime'] if 'Regime' in all_df else np.full(len(all_df), ''),
        ], axis=-1),
        hovertemplate=(
            'Date: %{text}<br>' +
            'S&P 500 MA % Change: %{customdata[0]:.2%}<br>' +
            'Inflation Rate MA Growth: %{customdata[1]:.2%}<br>' +
            'Regime: %{customdata[2]}<extra></extra>'
        ),
        name='All Data (Oldest to Newest)'
    ))
    all_scatter_fig.update_layout(
        xaxis_title='S&P 500 MA % Change',
        yaxis_title='Inflation Rate MA Growth',
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

    # --- Regime Periods Table ---
    st.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Periods</h2>
    """, unsafe_allow_html=True)
    # Prepare data for accurate regime periods: drop unknown or NaN
    df_periods = sp_inflation_data.dropna(subset=['Regime'])
    df_periods = df_periods[df_periods['Regime'] != 'Unknown']
    # Identify where each regime segment starts
    change_mask = df_periods['Regime'].ne(df_periods['Regime'].shift())
    df_start = df_periods.loc[change_mask, ['DateTime', 'Regime']].copy()
    # Format start dates
    df_start['Start Date'] = df_start['DateTime'].dt.strftime('%Y-%m-%d')
    # Compute end dates as next segment's start, last takes final date
    df_start['End Date'] = df_start['Start Date'].shift(-1)
    last_date = df_periods['DateTime'].iloc[-1].strftime('%Y-%m-%d')
    df_start.at[df_start.index[-1], 'End Date'] = last_date
    # Map regime numbers to labels
    df_start['Regime'] = df_start['Regime'].map(regime_labels_dict)
    # Final periods table
    periods_df = df_start[['Regime', 'Start Date', 'End Date']]
    periods_df = periods_df.sort_values('Start Date', ascending=False).reset_index(drop=True)
    # Highlight function for regime periods
    def highlight_regime_period(row):
        lbl = row['Regime']
        num = next((k for k,v in regime_labels_dict.items() if v==lbl), None)
        css = regime_bg_colors.get(num, '#eeeeee')
        if css.startswith('rgba'):
            m = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css)
            if m:
                r,g,b,_ = m.groups()
                from config.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})"
        else:
            color = '#eeeeee'
        return [f'background-color: {color}']*len(row)
    st.dataframe(
        periods_df.style.apply(highlight_regime_period, axis=1),
        use_container_width=True
    )

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

def generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list, regime_labels_dict):
    """
    Generate a trade log DataFrame for the given assets using regime periods,
    skipping any regime that starts before an asset has data.
    """
    # Identify regime periods
    dfp = sp_inflation_data.dropna(subset=['Regime'])
    dfp = dfp[dfp['Regime'] != 'Unknown']
    change = dfp['Regime'].ne(dfp['Regime'].shift())
    periods = dfp.loc[change, ['DateTime', 'Regime']].copy()
    periods['Start'] = periods['DateTime']
    periods['End'] = periods['Start'].shift(-1)
    periods.at[periods.index[-1], 'End'] = dfp['DateTime'].iloc[-1]
    periods['RegimeLabel'] = periods['Regime'].map(regime_labels_dict)
    results = []
    # Loop each regime period
    for row in periods.itertuples(index=False):
        start, end, regime_lbl = row.Start, row.End, row.RegimeLabel
        for asset in asset_list:
            df_a = merged_asset_data_metrics[['DateTime', asset]].dropna()
            if df_a.empty:
                continue
            first_date = df_a['DateTime'].min()
            # skip if regime starts before asset data begins
            if start < first_date:
                continue
            seg = df_a[(df_a['DateTime'] >= start) & (df_a['DateTime'] <= end)]
            if seg.empty:
                continue
            p0, p1 = seg[asset].iloc[0], seg[asset].iloc[-1]
            ret = (p1 - p0) / p0 if p0 else np.nan
            r = seg[asset].pct_change().dropna()
            vol = r.std() * np.sqrt(12) if not r.empty else np.nan
            cum = (1 + r).cumprod()
            dd = cum / cum.cummax() - 1
            mdd = dd.min() if not dd.empty else np.nan
            shp = ret / vol if vol and not np.isnan(ret) else np.nan
            results.append({
                'Asset': asset,
                'Regime': regime_lbl,
                'Start Date': start.strftime('%Y-%m-%d'),
                'End Date': end.strftime('%Y-%m-%d'),
                'Start Price': p0,
                'End Price': p1,
                'Period Return': ret,
                'Volatility': vol,
                'Sharpe Ratio': shp,
                'Max Drawdown': mdd
            })
    df_tl = pd.DataFrame(results)
    if df_tl.empty:
        return df_tl
    # Ordering
    order_regime = [regime_labels_dict[k] for k in [2,1,4,3]]
    df_tl['Regime'] = pd.Categorical(df_tl['Regime'], categories=order_regime, ordered=True)
    df_tl['Asset'] = pd.Categorical(df_tl['Asset'], categories=asset_list, ordered=True)
    df_tl = df_tl.sort_values(['Start Date', 'Regime', 'Asset'], ascending=[False, True, True]).reset_index(drop=True)
    return df_tl

def render_asset_analysis_tab(tab, title, asset_list, asset_colors, regime_bg_colors, regime_labels_dict, sp_inflation_data, asset_ts_data):
    tab.markdown(f"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>{title}</h2>
    """, unsafe_allow_html=True)
    with st.spinner('Merging asset data with regimes...'):
        merged_asset_data = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    # Compute regime_periods for consistent chart shading and trade log
    df_periods = sp_inflation_data.dropna(subset=['Regime'])
    df_periods = df_periods[df_periods['Regime'] != 'Unknown']
    change_mask2 = df_periods['Regime'].ne(df_periods['Regime'].shift())
    df_start2 = df_periods.loc[change_mask2, ['DateTime', 'Regime']].copy()
    # Format start dates
    df_start2['Start'] = df_start2['DateTime']
    df_start2['End'] = df_start2['Start'].shift(-1)
    df_start2.at[df_start2.index[-1], 'End'] = df_periods['DateTime'].iloc[-1]
    df_start2['Regime'] = df_start2['Regime'].map(regime_labels_dict)
    regime_periods = df_start2[['Regime', 'Start', 'End']].to_dict(orient='records')
    plot_asset_performance_over_time(
        merged_asset_data,
        asset_list,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        title + ' (Normalized to 100 at First Available Date)',
        regime_periods=regime_periods
    )
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Trade Log</h2>
    """, unsafe_allow_html=True)
    merged_asset_data_metrics = merged_asset_data.copy()
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list, regime_labels_dict)
    def highlight_regime(row):
        regime_label = row['Regime']
        regime_num = None
        for k, v in regime_labels_dict.items():
            if v == regime_label:
                regime_num = k
                break
        css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
        if css_rgba.startswith('rgba'):
            m = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
            if m:
                r,g,b,_ = m.groups()
                from config.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})"
        else:
            color = '#eeeeee'
        return [f'background-color: {color}'] * len(row)
    tab.dataframe(
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
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Aggregated Performance Metrics</h2>
    """, unsafe_allow_html=True)
    avg_metrics_table = generate_aggregated_metrics(trade_log_df, merged_asset_data_metrics, asset_list, regime_labels_dict)
    avg_metrics_table = avg_metrics_table[avg_metrics_table['Regime'] != 'Unknown'].reset_index(drop=True)
    regime_order = [regime_labels_dict[k] for k in [2,1,4,3]]
    asset_order = asset_list
    avg_metrics_table['Regime'] = pd.Categorical(avg_metrics_table['Regime'], categories=regime_order, ordered=True)
    avg_metrics_table['Asset'] = pd.Categorical(avg_metrics_table['Asset'], categories=asset_order, ordered=True)
    avg_metrics_table = avg_metrics_table.sort_values(['Regime','Asset']).reset_index(drop=True)
    def highlight_regime_avg(row):
        regime_label = row['Regime']
        regime_num = None
        for k, v in regime_labels_dict.items():
            if v == regime_label:
                regime_num = k
                break
        css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
        if css_rgba.startswith('rgba'):
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
            if match:
                r, g, b, _ = match.groups()
                from config.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = 'rgba(200,200,200,0.13)'
        else:
            color = '#eeeeee'
        return [f'background-color: {color}'] * len(row)
    tab.dataframe(
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
    plot_metrics_bar_charts(avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict)

# Tab 2: Asset Classes (Refactored)
with tabs[1]:
    render_asset_analysis_tab(
        tabs[1],
        "Asset Class Performance Over Time",
        asset_list_tab2,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data
    )

# Tab 3: Large vs. Small Cap
tabs[2].title = "Large vs. Small Cap"
with tabs[2]:
    render_asset_analysis_tab(
        tabs[2],
        "Performance of Large, Mid, Small, and Micro Cap Stocks Across Regimes",
        asset_list_tab3,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data
    )

# Tab 4: Cyclical vs. Defensive
tabs[3].title = "Cyclical vs. Defensive"
with tabs[3]:
    render_asset_analysis_tab(
        tabs[3],
        "Performance of MSCI USA Cyclical vs. Defensive Stocks Across Regimes",
        asset_list_tab4,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data
    )

# Tab 5: US Sectors
tabs[4].title = "US Sectors"
with tabs[4]:
    render_asset_analysis_tab(
        tabs[4],
        "Performance of US Sector ETFs Across Regimes",
        asset_list_tab5,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data
    )

# Tab 6: Factor Investing
tabs[5].title = "Factor Investing"
with tabs[5]:
    render_asset_analysis_tab(
        tabs[5],
        "Performance of MSCI World Factor Strategies Across Regimes",
        asset_list_tab6,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data
    )
