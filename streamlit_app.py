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
import tabs.all_weather  # New import
import tabs.us_sectors  # New import
import tabs.factor_investing  # New import
import tabs.large_vs_small  # New import
import tabs.cyclical_vs_defensive  # New import

from data.fetch import fetch_and_decode, decode_base64_data
from data.processing import merge_asset_with_regimes, compute_moving_average, compute_growth, assign_regimes
from viz.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from metrics.performance import generate_aggregated_metrics
from config.constants import asset_colors, regime_bg_colors, regime_legend_colors, regime_labels_dict, asset_list_tab2, asset_list_tab3, asset_list_tab4, asset_list_tab5, asset_list_tab6, asset_list_tab7, regime_definitions, REGIME_BG_ALPHA
from data.loader import load_data
from ui.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab

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

# Global checkbox for asset start override
if 'include_late_assets' not in st.session_state:
    st.session_state['include_late_assets'] = False
include_late_assets = st.session_state['include_late_assets']

# --- User-defined Start and End Dates BEFORE computing MAs and Growth ---
# Ensure sp_inflation_data and asset_ts_data are loaded before this block
try:
    min_date = min(sp_inflation_data['DateTime'].min(), asset_ts_data['DateTime'].min()).date()
    max_date = max(sp_inflation_data['DateTime'].max(), asset_ts_data['DateTime'].max()).date()
except Exception:
    with st.spinner('Loading data...'):
        sp_inflation_data, asset_ts_data = load_data()
    min_date = min(sp_inflation_data['DateTime'].min(), asset_ts_data['DateTime'].min()).date()
    max_date = max(sp_inflation_data['DateTime'].max(), asset_ts_data['DateTime'].max()).date()

# Separate sidebar inputs
start_date = st.sidebar.date_input(
    "Start Date", value=datetime.date(1972, 1, 31), min_value=min_date, max_value=max_date, key="start_date"
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
    # Compute moving-average-derived cutoff date
    ma_start_date = sp_inflation_data.loc[
        sp_inflation_data['S&P 500 MA'].notna(), 'DateTime'
    ].min().date()
    st.session_state['ma_start_date'] = ma_start_date
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
tab_objs = st.tabs(["Regime Visualization", "Asset Classes", "Large vs. Small Cap", "Cyclical vs. Defensive", "US Sectors", "Factor Investing", "All-Weather Portfolio"])
t1 = time.time()
print(f"DEBUG: Tab setup took {t1-t0:.2f} seconds.")

# Tab 1: Regime Visualization
with tab_objs[0]:
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
        1: "🎈 <b>Reflation</b>: Rising growth, rising inflation",
        2: "👧🏼 <b>Goldilocks</b>: Rising growth, falling inflation",
        3: "✋ <b>Stagflation</b>: Falling growth, rising inflation",
        4: "💨 <b>Deflation</b>: Falling growth, falling inflation"
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
        fillcolor=regime_bg_colors[1],  # Reflation (red)
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max, y0=y_bg_min, y1=0,
        fillcolor=regime_bg_colors[2],  # Goldilocks (green)
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min, x1=0, y0=0, y1=y_bg_max,
        fillcolor=regime_bg_colors[3],  # Stagflation (yellow)
        line_width=0, layer="below"
    )
    scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min, x1=0, y0=y_bg_min, y1=0,
        fillcolor=regime_bg_colors[4],  # Deflation (blue)
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
        fillcolor=regime_bg_colors[1],  # Reflation (red)
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=0, x1=x_bg_max_all, y0=y_bg_min_all, y1=0,
        fillcolor=regime_bg_colors[2],  # Goldilocks (green)
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min_all, x1=0, y0=0, y1=y_bg_max_all,
        fillcolor=regime_bg_colors[3],  # Stagflation (yellow)
        line_width=0, layer="below"
    )
    all_scatter_fig.add_shape(
        type="rect",
        x0=x_bg_min_all, x1=0, y0=y_bg_min_all, y1=0,
        fillcolor=regime_bg_colors[4],  # Deflation (blue)
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
    dfp = sp_inflation_data.dropna(subset=['Regime'])
    dfp = dfp[dfp['Regime'] != 'Unknown']
    change_mask = dfp['Regime'].ne(dfp['Regime'].shift())
    df_start = dfp.loc[change_mask, ['DateTime', 'Regime']].copy()
    # Format start dates
    df_start['Start Date'] = df_start['DateTime'].dt.strftime('%Y-%m-%d')
    # Compute end dates as next segment's start, last takes final date
    df_start['End Date'] = df_start['Start Date'].shift(-1)
    last_date = dfp['DateTime'].iloc[-1].strftime('%Y-%m-%d')
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

def render_asset_analysis_tab(tab, title, asset_list, asset_colors, regime_bg_colors, regime_labels_dict, sp_inflation_data, asset_ts_data, include_pre_cutoff_trades=False, include_late_assets=False, cutoff_date=None, eligible_assets=None, tab_title=""):
    tab.markdown(f"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>{title}</h2>
    """, unsafe_allow_html=True)
    # Use the passed include_late_assets argument in filtering logic
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
    # Set xaxis range for normalized asset charts if tab-specific cutoff is used
    xaxis_range = None
    from config.constants import asset_list_tab3, asset_list_tab6
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
        title + ' (Normalized to 100 at First Available Date)',
        regime_periods=regime_periods,
        xaxis_range=xaxis_range
    )
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Trade Log</h2>
    """, unsafe_allow_html=True)
    merged_asset_data_metrics = merged_asset_data.copy()
    from metrics.performance import generate_trade_log_df
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list, regime_labels_dict)
    # --- Asset filtering for aggregated metrics/bar charts ---
    # Compute first available date for each asset
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list if asset in asset_ts_data.columns
    }
    print(f"[DEBUG] asset_first_date for tab '{title}':")
    for asset, date in asset_first_date.items():
        print(f"    {asset}: {date}")
    # Use the tab-specific include_late_assets value
    passed_cutoff_date = cutoff_date # Rename argument to avoid confusion
    from config.constants import asset_list_tab3 # Remove asset_list_tab6 import here

    if passed_cutoff_date is not None:
        cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else st.session_state.get('ma_start_date')
    elif asset_list == asset_list_tab3:
        cutoff_date = datetime.date(1994, 6, 30) # Hardcoded for Tab 3 (Large vs Small)
        print(f"[DEBUG] Using hardcoded cutoff_date for tab '{title}': {cutoff_date}")
    # REMOVED: elif asset_list == asset_list_tab6: condition
    else:
        cutoff_date = st.session_state.get('ma_start_date') # Fallback to MA start date
        print(f"[DEBUG] Using fallback cutoff_date (ma_start_date) for tab '{title}': {cutoff_date}")

    print(f"[DEBUG] Final cutoff_date being used for filtering in tab '{title}': {cutoff_date}")
    print(f"[DEBUG] include_late_assets for tab '{title}': {include_late_assets}")

    if not include_late_assets and cutoff_date is not None:
        eligible_assets = [a for a, d in asset_first_date.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_first_date:
                min_date = min(asset_first_date.values())
                eligible_assets = [a for a, d in asset_first_date.items() if d == min_date]
            else:
                eligible_assets = []
    else:
        eligible_assets = [a for a in asset_list if a in asset_ts_data.columns]
    print(f"[DEBUG] eligible_assets for tab '{title}': {eligible_assets}")

    # --- Central eligibility function for trade inclusion ---
    def is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override):
        asset = row.get('Asset', None)
        trade_start_date = row.get('Start Date', None)
        # Try to parse trade_start_date as date if it's a string
        if isinstance(trade_start_date, str):
            try:
                trade_start_date = pd.to_datetime(trade_start_date).date()
            except Exception:
                trade_start_date = None
        if asset not in eligible_assets:
            return False
        if trade_start_date is not None and trade_start_date < cutoff_date and not pre_cutoff_override:
            return False
        return True

    # Determine pre_cutoff_override for this tab
    pre_cutoff_override = include_pre_cutoff_trades

    # --- AGGREGATION: filter trades for metrics/bar charts based on eligibility function ---
    if 'Asset' in trade_log_df:
        trade_log_df = trade_log_df[trade_log_df['Asset'].isin(asset_list)].copy()
    else:
        trade_log_df = trade_log_df.copy()
    eligible_mask = trade_log_df.apply(lambda row: is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override), axis=1)
    filtered_trade_log_df = trade_log_df[eligible_mask].copy()

    # --- Trade log highlighting uses the same function ---
    def highlight_trade(row):
        if not is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override):
            return ['background-color: #e0e0e0'] * len(row)
        # Otherwise, use regime background color
        regime_label = row.get('Regime', None)
        regime_num = None
        for k, v in regime_labels_dict.items():
            if v == regime_label:
                regime_num = k
                break
        css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
        if css_rgba.startswith('rgba'):
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
            if match:
                r,g,b,_ = match.groups()
                from config.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})" # Fallback
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
            .apply(highlight_trade, axis=1),
        use_container_width=True
    )
    # --- FOOTNOTES for Trade Log ---
    tab.markdown("""
**Column Definitions:**
- **Period Return**: (End Price - Start Price) / Start Price
- **Volatility**: Standard deviation of monthly returns within the period, annualized (multiplied by $\sqrt{12}$)
- **Sharpe Ratio**: Annualized mean monthly return divided by annualized volatility, assuming risk-free rate = 0
- **Max Drawdown**: Maximum observed loss from a peak to a trough during the period, based on monthly closing prices (as a percentage of the peak)

*Volatility and Sharpe ratio cannot be calculated for 1-month periods.*
""", unsafe_allow_html=True)

    # Keep conditional footnote from upstream
    from config.constants import asset_list_tab3, asset_list_tab5, asset_list_tab6
    if asset_list in [asset_list_tab3, asset_list_tab5, asset_list_tab6]:
        tab.markdown(
            '*If the background color is gray, the trade is not included in the aggregations and the bar charts.*',
            unsafe_allow_html=True
        )

    # --- AGGREGATED METRICS TABLE --- (Keep title, use upstream logic for avg_metrics_table)
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Aggregated Performance Metrics</h2>
    """, unsafe_allow_html=True)

    # Use eligible_assets for aggregated metrics and bar charts (from upstream)
    avg_metrics_table = generate_aggregated_metrics(filtered_trade_log_df, merged_asset_data_metrics, eligible_assets, regime_labels_dict)
    avg_metrics_table = avg_metrics_table[avg_metrics_table['Regime'] != 'Unknown']
    avg_metrics_table = avg_metrics_table[avg_metrics_table['Asset'].isin(eligible_assets)]
    avg_metrics_table = avg_metrics_table.reset_index(drop=True)
    regime_order = [regime_labels_dict[k] for k in [2,1,4,3]]
    asset_order = eligible_assets # Use eligible assets for ordering

    # Keep common formatting/ordering logic
    avg_metrics_table['Regime'] = pd.Categorical(avg_metrics_table['Regime'], categories=regime_order, ordered=True)
    avg_metrics_table['Asset'] = pd.Categorical(avg_metrics_table['Asset'], categories=asset_order, ordered=True)
    avg_metrics_table = avg_metrics_table.sort_values(['Regime','Asset']).reset_index(drop=True)

    # Display the formatted table
    def highlight_regime_avg(row):
        regime_label = row['Regime']
        regime_num = next((k for k, v in regime_labels_dict.items() if v == regime_label), None)
        css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
        # Use REGIME_BG_ALPHA for consistent background intensity
        if css_rgba.startswith('rgba'):
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
            if match:
                r,g,b,_ = match.groups()
                from config.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})" # Fallback
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

    # --- FOOTNOTES for Aggregated Performance Table ---
    tab.markdown("""
**Aggregation & Calculation Notes:**
- **Annualized Return (Aggregated):** Average of monthly returns for each regime-asset group, annualized by multiplying by 12.
- **Annualized Volatility (Aggregated):** Standard deviation of those monthly returns, annualized by multiplying by √12.
- **Sharpe Ratio (Aggregated):** Aggregated annual return divided by aggregated annual volatility (0% risk-free rate).
- **Average Max Drawdown:** Mean of the maximum drawdowns observed in each period for each regime-asset group.
- **Missing Data Handling:** Excludes any missing (NaN) values from all calculations.



""", unsafe_allow_html=True)
    plot_metrics_bar_charts(avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict, tab_title)

# Tab 2: Asset Classes (Refactored)
with tab_objs[1]:
    render_asset_analysis_tab(
        tab_objs[1],
        "Asset Class Performance Over Time",
        asset_list_tab2,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        "Asset Classes"
    )

# Tab 6: Factor Investing
with tab_objs[5]:
    tabs.factor_investing.render(
        tab_objs[5],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state
    )

# Tab 3: Large vs. Small Cap
with tab_objs[2]:
    tabs.large_vs_small.render(
        tab_objs[2],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state
    )

# Tab 4: Cyclical vs. Defensive
with tab_objs[3]:
    tabs.cyclical_vs_defensive.render(
        tab_objs[3],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state
    )

# Tab 5: US Sectors
with tab_objs[4]:
    import tabs.us_sectors
    tabs.us_sectors.render(
        tab_objs[4],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state
    )

# Tab 7: All-Weather Portfolio
with tab_objs[6]:
    tabs.all_weather.render(
        tab_objs[6],
        asset_ts_data=asset_ts_data,
        sp_inflation_data=sp_inflation_data,
        session_state=st.session_state
    )
