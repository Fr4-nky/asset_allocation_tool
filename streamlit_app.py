# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import re
import concurrent.futures
import datetime  # for default dates and cutoffs
import hashlib
import logging
import os
import tabs.all_weather  # New import
import tabs.us_sectors  # New import
import tabs.factor_investing  # New import
import tabs.large_vs_small  # New import
import tabs.cyclical_vs_defensive  # New import
import tabs.asset_classes  # New import
import tabs.regime  # New import
import requests  # For making API calls
from urllib.parse import quote  # For URL encoding

from core.fetch import fetch_and_decode, decode_base64_data
from core.processing import (
    merge_asset_with_regimes,
    compute_moving_average,
    compute_growth,
    assign_regimes,
)
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.performance import generate_aggregated_metrics
from core.constants import (
    asset_colors,
    regime_bg_colors,
    regime_legend_colors,
    regime_labels_dict,
    asset_list_tab2,
    asset_list_tab3,
    asset_list_tab4,
    asset_list_tab5,
    asset_list_tab6,
    asset_list_tab7,
    regime_definitions,
    REGIME_BG_ALPHA,
)
from core.loader import load_data
from core.asset_analysis import (
    get_dynamic_cutoff_date_from_trade_log,
    render_asset_analysis_tab,
)


# --- User Authentication ---
from config import API_BASE_URL, DEBUG

# Configure logging only for authentication
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a specific logger for authentication
auth_logger = logging.getLogger("auth")
auth_logger.setLevel(logging.INFO)

# Create file handler for authentication logs
auth_handler = logging.FileHandler(os.path.join(log_dir, "auth.log"))
auth_handler.setLevel(logging.INFO)

# Create formatter
auth_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
auth_handler.setFormatter(auth_formatter)

# Add handler to logger
if not auth_logger.handlers:
    auth_logger.addHandler(auth_handler)

query_params = st.query_params

email = query_params["email"] if "email" in query_params else None
auth_logger.info(f"User email from query params: {email}")

# If DEBUG is true, allow all emails to access premium content
if DEBUG:
    st.session_state.is_premium_user = True
    auth_logger.info(
        f"DEBUG mode enabled - setting premium access to True for email: {email}"
    )
else:
    # URL encode the email to handle special characters like +
    encoded_email = quote(email) if email else ""
    verify_endpoint = (
        f"{API_BASE_URL}/community/verify-user-membership/?email={encoded_email}"
        if API_BASE_URL
        else f"http://localhost:8000/community/verify-user-membership?email={encoded_email}"
    )
    auth_logger.info(f"Original email: {email}")
    auth_logger.info(f"URL encoded email: {encoded_email}")
    auth_logger.info(f"Making API call to verify user membership: {verify_endpoint}")

    try:
        
        response = requests.get(verify_endpoint, verify=False)
        auth_logger.info(f"API response status: {response.status_code}")
        auth_logger.debug(f"API response content: {response.text}")

        response_data = response.json()
        st.session_state.is_premium_user = response_data.get("is_premium_member", False)
        auth_logger.info(
            f"Extracted is_premium_member: {st.session_state.is_premium_user}"
        )

    except Exception as e:
        auth_logger.error(f"Error during membership verification: {str(e)}")
        st.session_state.is_premium_user = False
        auth_logger.info("Setting premium access to False due to error")

auth_logger.info(
    f"Final user membership status - Email: {email}, Is Premium: {st.session_state.is_premium_user}"
)


# Set page configuration
st.set_page_config(
    page_title="Macroeconomic Regimes and Asset Performance",
    layout="wide",
    page_icon="https://www.longtermtrends.net/static/my_app/images/favicon.ico",
)

# --- Add meta tags for SEO and social sharing ---
meta_title = "Macroeconomic Regimes and Asset Performance Analysis | Longtermtrends"
meta_description = "This app visualizes macroeconomic regimes based on S&P 500 and Inflation Rate data, and analyzes asset performance across these regimes."
# Inject meta tags into the Streamlit app (HTML head)
st.markdown(
    f"""
    <meta name='title' content='{meta_title}'>
    <meta name='description' content='{meta_description}'>
    <meta property='og:title' content='{meta_title}'>
    <meta property='og:description' content='{meta_description}'>
    <meta property='og:type' content='website'>
""",
    unsafe_allow_html=True,
)

# Title and Description
st.title("Macroeconomic Regimes and Asset Performance Analysis")
st.write(
    """
_Assessing trends in economic growth and inflation to define distinct market regimes—such as Reflation, Goldilocks, Stagflation, or Deflation/Disinflation—is a widely used approach to asset allocation. Understanding how assets behave in these environments offers practical insights for portfolio design and risk management._

This tool is designed to support more robust portfolio construction: all members can access regime visualizations and trade logs with performance metrics for each asset and regime, while aggregated performance metrics are available to premium members.
"""
)

# Sidebar User Inputs
st.sidebar.header("User Input")
# Single moving average input for both S&P 500 and Inflation
ma_length = st.sidebar.number_input(
    "Moving Average Length (Months)",
    min_value=1,
    max_value=24,
    value=12,
    step=1,
    key="ma_length",
    help="Number of months for the moving average window for both S&P 500 and Inflation.",
)
sp500_n = ma_length
inflation_n = ma_length

# Global checkbox for asset start override
if "include_late_assets" not in st.session_state:
    st.session_state["include_late_assets"] = False
include_late_assets = st.session_state["include_late_assets"]

# Checkbox for S&P 500 inflation adjustment in the sidebar
if 'adjust_sp500_for_inflation' not in st.session_state:
    st.session_state.adjust_sp500_for_inflation = False # Default to False

st.session_state.adjust_sp500_for_inflation = st.sidebar.checkbox(
    "Adjust S&P 500 for inflation (for regime calculation)",
    value=st.session_state.adjust_sp500_for_inflation,
    key='adjust_sp500_for_inflation_sidebar_checkbox',
    help="If checked, the S&P 500 series used for regime definition will be divided by CPI to get real values."
)


# Data Loading & Caching
if "sp_inflation_data" not in st.session_state:
    with st.spinner("Loading data..."):
        sp_inflation_data, asset_ts_data = load_data()
        st.session_state["sp_inflation_data"] = sp_inflation_data
        st.session_state["asset_ts_data"] = asset_ts_data
sp_inflation_data = st.session_state["sp_inflation_data"]
asset_ts_data = st.session_state["asset_ts_data"]

# Determine date range
min_date = min(
    sp_inflation_data["DateTime"].min(), asset_ts_data["DateTime"].min()
).date()
max_date = max(
    sp_inflation_data["DateTime"].max(), asset_ts_data["DateTime"].max()
).date()

# Separate sidebar inputs
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(1972, 1, 31),
    min_value=min_date,
    max_value=max_date,
    key="start_date",
)
end_date = st.sidebar.date_input(
    "End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date"
)
if start_date > end_date:
    st.sidebar.error("Start Date must be on or before End Date")
    st.stop()
# Debug logging

# Convert to Timestamps
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
# Filter both datasets
sp_inflation_data = sp_inflation_data[
    (sp_inflation_data["DateTime"] >= start_date)
    & (sp_inflation_data["DateTime"] <= end_date)
].copy()
asset_ts_data = asset_ts_data[
    (asset_ts_data["DateTime"] >= start_date) & (asset_ts_data["DateTime"] <= end_date)
].copy()


# --- Apply S&P 500 Inflation Adjustment if selected ---
# The key used here must match the key of the checkbox in the sidebar
if st.session_state.get('adjust_sp500_for_inflation_sidebar_checkbox', False):
    if 'CPI' in sp_inflation_data.columns and not sp_inflation_data['CPI'].dropna().empty: # Check if there is ANY non-NaN CPI data
        if 'S&P 500' in sp_inflation_data.columns and not sp_inflation_data['S&P 500'].isnull().all():
            # Forward-fill the CPI series. This uses the last known CPI for recent S&P 500 dates
            # where current CPI might be NaN due to data lag.
            cpi_series_filled = sp_inflation_data['CPI'].ffill()
            # After forward-filling, replace any 0s with NaN to avoid division by zero.
            # This is a safeguard in case 0 is a valid reported CPI value (unlikely for broad indices) or if ffill propagates a 0.
            cpi_series_filled = cpi_series_filled.replace(0, np.nan)

            # Check if the forward-filled CPI series has any valid (non-NaN) data to use for adjustment.
            if not cpi_series_filled.dropna().empty:
                sp_inflation_data['S&P 500 Original (Nominal)'] = sp_inflation_data['S&P 500'].copy()
                sp_inflation_data['S&P 500'] = sp_inflation_data['S&P 500'] / cpi_series_filled

            else:
                st.warning("Cannot adjust S&P 500 for inflation: Forward-filled CPI series resulted in no valid data (e.g., all zeros or NaNs). Using unadjusted S&P 500.")
        else:
            st.warning("Cannot adjust S&P 500 for inflation: 'S&P 500' data is missing or empty. Using unadjusted S&P 500.")
    else:
        st.warning("Cannot adjust S&P 500 for inflation: 'CPI' data is missing, empty, or contains no valid numbers. Using unadjusted S&P 500.")
else:
    # Ensure the original S&P 500 is used if adjustment is not selected or was previously applied
    if 'S&P 500 Original (Nominal)' in sp_inflation_data.columns:
        sp_inflation_data['S&P 500'] = sp_inflation_data['S&P 500 Original (Nominal)'].copy()
        # Optionally remove the temporary column if no longer needed, 
        # but it's good to keep it if the user might toggle the checkbox back and forth.
        # del sp_inflation_data['S&P 500 Original (Nominal)'] 



# --- Logging for Moving Average Computation ---
t0 = time.time()
with st.spinner("Computing Moving Averages..."):
    sp_inflation_data["S&P 500 MA"] = compute_moving_average(
        sp_inflation_data["S&P 500"], window_size=sp500_n
    )
    sp_inflation_data["Inflation Rate MA"] = compute_moving_average(
        sp_inflation_data["Inflation Rate"], window_size=inflation_n
    )

    # Compute moving-average-derived cutoff date
    ma_start_date = (
        sp_inflation_data.loc[sp_inflation_data["S&P 500 MA"].notna(), "DateTime"]
        .min()
        .date()
    )
    st.session_state["ma_start_date"] = ma_start_date
t1 = time.time()


# --- Logging for Growth Computation ---
t0 = time.time()
with st.spinner("Computing Growth..."):
    sp_inflation_data["S&P 500 MA Growth"] = compute_growth(
        sp_inflation_data["S&P 500 MA"]
    )
    sp_inflation_data["Inflation Rate MA Growth"] = compute_growth(
        sp_inflation_data["Inflation Rate MA"]
    )

    # --- DEBUG PRINTS FOR GROWTH COLUMNS ---




    # --- DO NOT DROP ROWS WITH NANs IN MA OR GROWTH COLUMNS BEFORE REGIME ASSIGNMENT OR PLOTTING (MATCH OLD BEHAVIOR) ---
    # sp_inflation_data = sp_inflation_data.dropna(subset=[
    #     'S&P 500 MA', 'Inflation Rate MA', 'S&P 500 MA Growth', 'Inflation Rate MA Growth'
    # ]).copy()
    # print("DEBUG: After dropna, sp_inflation_data shape:", sp_inflation_data.shape)
t1 = time.time()


# Now that we have the growth, we can get min and max values
sp500_growth = sp_inflation_data["S&P 500 MA Growth"].dropna()
inflation_growth = sp_inflation_data["Inflation Rate MA Growth"].dropna()

sp500_min = float(sp500_growth.min())
sp500_max = float(sp500_growth.max())
inflation_min = float(inflation_growth.min())
inflation_max = float(inflation_growth.max())

# --- Logging for Regime Assignment ---
t0 = time.time()
with st.spinner("Assigning Regimes..."):
    sp_inflation_data = assign_regimes(sp_inflation_data, regime_definitions)

t1 = time.time()


# Handle any NaN regimes (should not happen)
sp_inflation_data["Regime"] = sp_inflation_data["Regime"].fillna("Unknown")

# --- Logging for Tab Rendering ---
t0 = time.time()

tab_objs = st.tabs(
    [
        "Regime Visualization",
        "Asset Classes",
        "Large vs. Small Cap",
        "Cyclical vs. Defensive",
        "US Sectors",
        "Factor Investing",
        "All-Weather Portfolio",
    ]
)
t1 = time.time()


# Tab 1: Regime Visualization
import tabs.regime

with tab_objs[0]:
    tabs.regime.render(tab_objs[0], sp_inflation_data)


def render_asset_analysis_tab(
    tab,
    title,
    asset_list,
    asset_colors,
    regime_bg_colors,
    regime_labels_dict,
    sp_inflation_data,
    asset_ts_data,
    include_pre_cutoff_trades=False,
    include_late_assets=False,
    cutoff_date=None,
    eligible_assets=None,
    tab_title="",
):
    tab.markdown(
        r"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>{title}</h2>
    """,
        unsafe_allow_html=True,
    )
    # Use the passed include_late_assets argument in filtering logic
    with st.spinner("Merging asset data with regimes..."):
        merged_asset_data = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    # Compute regime_periods for consistent chart shading and trade log
    df_periods = sp_inflation_data.dropna(subset=["Regime"])
    df_periods = df_periods[df_periods["Regime"] != "Unknown"]
    change_mask2 = df_periods["Regime"].ne(df_periods["Regime"].shift())
    df_start2 = df_periods.loc[change_mask2, ["DateTime", "Regime"]].copy()
    # Format start dates
    df_start2["Start"] = df_start2["DateTime"]
    df_start2["End"] = df_start2["Start"].shift(-1)
    df_start2.at[df_start2.index[-1], "End"] = df_periods["DateTime"].iloc[-1]
    df_start2["Regime"] = df_start2["Regime"].map(regime_labels_dict)
    regime_periods = df_start2[["Regime", "Start", "End"]].to_dict(orient="records")
    # Set xaxis range for normalized asset charts if tab-specific cutoff is used
    xaxis_range = None
    from core.constants import asset_list_tab3, asset_list_tab6

    if asset_list == asset_list_tab3:
        xaxis_range = ["1994-06-30", None]
    elif asset_list == asset_list_tab6:
        xaxis_range = ["1977-03-31", None]
    plot_asset_performance_over_time(
        merged_asset_data,
        asset_list,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        title + " (Normalized to 100 at First Available Date)",
        regime_periods=regime_periods,
        xaxis_range=xaxis_range,
    )
    tab.markdown(
        r"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Trade Log</h2>
    """,
        unsafe_allow_html=True,
    )
    merged_asset_data_metrics = merged_asset_data.copy()
    from core.performance import generate_trade_log_df

    trade_log_df = generate_trade_log_df(
        merged_asset_data_metrics, sp_inflation_data, asset_list, regime_labels_dict
    )
    # --- Asset filtering for aggregated metrics/bar charts ---
    # Compute first available date for each asset
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), "DateTime"].min().date()
        for asset in asset_list
        if asset in asset_ts_data.columns
    }

    # Use the tab-specific include_late_assets value

    if passed_cutoff_date is not None:
        cutoff_date = (
            dynamic_cutoff_date
            if dynamic_cutoff_date is not None
            else st.session_state.get("ma_start_date")
        )
    elif asset_list == asset_list_tab3:
        cutoff_date = datetime.date(1994, 6, 30)  # Hardcoded for Tab 3 (Large vs Small)

    # REMOVED: elif asset_list == asset_list_tab6: condition
    else:
        cutoff_date = st.session_state.get("ma_start_date")  # Fallback to MA start date





    if not include_late_assets and cutoff_date is not None:
        eligible_assets = [a for a, d in asset_first_date.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_first_date:
                min_date = min(asset_first_date.values())
                eligible_assets = [
                    a for a, d in asset_first_date.items() if d == min_date
                ]
            else:
                eligible_assets = []
    else:
        eligible_assets = [a for a in asset_list if a in asset_ts_data.columns]


    # --- Central eligibility function for trade inclusion ---
    def is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override):
        asset = row.get("Asset", None)
        trade_start_date = row.get("Start Date", None)
        # Try to parse trade_start_date as date if it's a string
        if isinstance(trade_start_date, str):
            try:
                trade_start_date = pd.to_datetime(trade_start_date).date()
            except Exception:
                trade_start_date = None
        if asset not in eligible_assets:
            return False
        if (
            trade_start_date is not None
            and trade_start_date < cutoff_date
            and not pre_cutoff_override
        ):
            return False
        return True

    # Determine pre_cutoff_override for this tab
    pre_cutoff_override = include_pre_cutoff_trades

    # --- AGGREGATION: filter trades for metrics/bar charts based on eligibility function ---
    if "Asset" in trade_log_df:
        trade_log_df = trade_log_df[trade_log_df["Asset"].isin(asset_list)].copy()
    else:
        trade_log_df = trade_log_df.copy()
    eligible_mask = trade_log_df.apply(
        lambda row: is_trade_eligible(
            row, eligible_assets, cutoff_date, pre_cutoff_override
        ),
        axis=1,
    )
    filtered_trade_log_df = trade_log_df[eligible_mask].copy()

    # --- Trade log highlighting uses the same function ---
    def highlight_trade(row):
        if not is_trade_eligible(
            row, eligible_assets, cutoff_date, pre_cutoff_override
        ):
            return ["background-color: #e0e0e0"] * len(row)
        # Otherwise, use regime background color
        regime_label = row.get("Regime", None)
        regime_num = None
        for k, v in regime_labels_dict.items():
            if v == regime_label:
                regime_num = k
                break
        css_rgba = regime_bg_colors.get(regime_num, "#eeeeee")
        if css_rgba.startswith("rgba"):
            match = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", css_rgba)
            if match:
                r, g, b, _ = match.groups()
                from core.constants import REGIME_BG_ALPHA

                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})"  # Fallback
        else:
            color = "#eeeeee"
        return [f"background-color: {color}"] * len(row)

    tab.dataframe(
        trade_log_df.style.format(
            {
                "Start Price": "{:.2f}",
                "End Price": "{:.2f}",
                "Period Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown": "{:.2%}",
            }
        ).apply(highlight_trade, axis=1),
        use_container_width=True,
    )
    # --- FOOTNOTES for Trade Log ---
    tab.markdown(
        r"""
**Column Definitions:**
- **Period Return**: (End Price - Start Price) / Start Price
- **Volatility**: Standard deviation of monthly returns within the period, annualized (multiplied by $\sqrt{12}$)
- **Sharpe Ratio**: Annualized mean monthly return divided by annualized volatility, assuming risk-free rate = 0
- **Max Drawdown**: Maximum observed loss from a peak to a trough during the period, based on monthly closing prices (as a percentage of the peak)

*Volatility and Sharpe ratio cannot be calculated for 1-month periods.*
""",
        unsafe_allow_html=True,
    )

    # Keep conditional footnote from upstream
    from core.constants import asset_list_tab3, asset_list_tab5, asset_list_tab6

    if asset_list in [asset_list_tab3, asset_list_tab5, asset_list_tab6]:
        tab.markdown(
            "*If the background color is gray, the trade is not included in the aggregations and the bar charts.*",
            unsafe_allow_html=True,
        )

    # --- AGGREGATED METRICS TABLE --- (Keep title, use upstream logic for avg_metrics_table)
    tab.markdown(
        r"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Aggregated Performance Metrics</h2>
    """,
        unsafe_allow_html=True,
    )

    # Use eligible_assets for aggregated metrics and bar charts (from upstream)
    avg_metrics_table = generate_aggregated_metrics(
        filtered_trade_log_df,
        merged_asset_data_metrics,
        eligible_assets,
        regime_labels_dict,
    )
    avg_metrics_table = avg_metrics_table[avg_metrics_table["Regime"] != "Unknown"]
    avg_metrics_table = avg_metrics_table[
        avg_metrics_table["Asset"].isin(eligible_assets)
    ]
    avg_metrics_table = avg_metrics_table.reset_index(drop=True)
    regime_order = [regime_labels_dict[k] for k in [2, 1, 4, 3]]
    asset_order = eligible_assets  # Use eligible assets for ordering

    # Keep common formatting/ordering logic
    avg_metrics_table["Regime"] = pd.Categorical(
        avg_metrics_table["Regime"], categories=regime_order, ordered=True
    )
    avg_metrics_table["Asset"] = pd.Categorical(
        avg_metrics_table["Asset"], categories=asset_order, ordered=True
    )
    avg_metrics_table = avg_metrics_table.sort_values(["Regime", "Asset"]).reset_index(
        drop=True
    )

    # Display the formatted table
    def highlight_regime_avg(row):
        regime_label = row["Regime"]
        regime_num = next(
            (k for k, v in regime_labels_dict.items() if v == regime_label), None
        )
        css_rgba = regime_bg_colors.get(regime_num, "#eeeeee")
        # Use REGIME_BG_ALPHA for consistent background intensity
        if css_rgba.startswith("rgba"):
            match = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", css_rgba)
            if match:
                r, g, b, _ = match.groups()
                from core.constants import REGIME_BG_ALPHA

                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})"  # Fallback
        else:
            color = "#eeeeee"
        return [f"background-color: {color}"] * len(row)

    tab.dataframe(
        avg_metrics_table.style.format(
            {
                "Annualized Return (Aggregated)": "{:.2%}",
                "Annualized Volatility (Aggregated)": "{:.2%}",
                "Sharpe Ratio (Aggregated)": "{:.2f}",
                "Average Max Drawdown (Period Avg)": "{:.2%}",
            }
        ).apply(highlight_regime_avg, axis=1),
        use_container_width=True,
    )

    # --- FOOTNOTES for Aggregated Performance Table ---
    tab.markdown(
        r"""
**Aggregation & Calculation Notes:**
- **Annualized Return (Aggregated):** Average of monthly returns for each regime-asset group, annualized by multiplying by 12.
- **Annualized Volatility (Aggregated):** Standard deviation of those monthly returns, annualized by multiplying by √12.
- **Sharpe Ratio (Aggregated):** Aggregated annual return divided by aggregated annual volatility (0% risk-free rate).
- **Average Max Drawdown:** Mean of the maximum drawdowns observed in each period for each regime-asset group.
- **Missing Data Handling:** Excludes any missing (NaN) values from all calculations.



""",
        unsafe_allow_html=True,
    )
    plot_metrics_bar_charts(
        avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict, tab_title
    )


# Tab 2: Asset Classes
import tabs.asset_classes

with tab_objs[1]:
    tabs.asset_classes.render(
        tab_objs[1],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )

# Tab 6: Factor Investing
with tab_objs[5]:
    tabs.factor_investing.render(
        tab_objs[5],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )

# Tab 3: Large vs. Small Cap
with tab_objs[2]:
    tabs.large_vs_small.render(
        tab_objs[2],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )

# Tab 4: Cyclical vs. Defensive
with tab_objs[3]:
    tabs.cyclical_vs_defensive.render(
        tab_objs[3],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )

# Tab 5: US Sectors
with tab_objs[4]:
    import tabs.us_sectors

    tabs.us_sectors.render(
        tab_objs[4],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )

# Tab 7: All-Weather Portfolio
with tab_objs[6]:
    tabs.all_weather.render(
        tab_objs[6],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state,
    )
