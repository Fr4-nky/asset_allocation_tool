# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io 

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
Select the moving average period, assets, and explore how regimes affect asset performance over time.
""")

# Sidebar User Inputs
st.sidebar.header("User Input Parameters")

# List of 'n' values in months
n_values = [int(i * 12) for i in [1/6, 1/4, 1/2, 3/4, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15]]
n = st.sidebar.select_slider("Select n (months) for Moving Average:", options=n_values, value=12)

# Load Data Function
@st.cache_data
def load_data():
    path = './processed_data/'
    # Load regime data
    regime_df = pd.read_csv(path + 'sp500_and_inflation_processed.csv', parse_dates=['DateTime'])
    # Load asset performance data
    asset_perf_df = pd.read_csv(path + 'assets_performance_by_regime.csv')
    # Load asset time series data
    asset_ts_df = pd.read_csv(path + 'asset_classes_preprocessed.csv', parse_dates=['DateTime'])
    return regime_df.copy(), asset_perf_df.copy(), asset_ts_df.copy()

with st.spinner('Loading data...'):
    regime_data, asset_perf_data, asset_ts_data = load_data()

# Ensure 'DateTime' is datetime type
regime_data['DateTime'] = pd.to_datetime(regime_data['DateTime'])
asset_ts_data['DateTime'] = pd.to_datetime(asset_ts_data['DateTime'])

# Data range selection
min_date = max(regime_data['DateTime'].min(), asset_ts_data['DateTime'].min())
max_date = min(regime_data['DateTime'].max(), asset_ts_data['DateTime'].max())

start_date = st.sidebar.date_input('Start date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End date', max_date, min_value=min_date, max_value=max_date)

# Convert start_date and end_date to pd.Timestamp
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date.')
    st.stop()

# Sidebar Asset Selection
asset_options = list(asset_ts_data.columns)
asset_options.remove('DateTime')
selected_assets = st.sidebar.multiselect("Select Assets to Display:", asset_options, default=['Gold', 'Bonds'])

# Sidebar Regime Selection
regime_options = [1, 2, 3, 4]
selected_regimes = st.sidebar.multiselect("Select Regimes to Include:", regime_options, default=regime_options)

# Sidebar Performance Metrics Selection
metric_options = ['Average Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
selected_metrics = st.sidebar.multiselect("Select Performance Metrics to Display:", metric_options, default=metric_options)

# Map regime numbers to colors and labels for visualization
regime_colors = {
    1: ('green', 'Rising S&P 500 & Rising Inflation Rate'),
    2: ('yellow', 'Rising S&P 500 & Falling Inflation Rate'),
    3: ('orange', 'Falling S&P 500 & Rising Inflation Rate'),
    4: ('red', 'Falling S&P 500 & Falling Inflation Rate')
}

# Caching filtered and merged data
@st.cache_data
def get_filtered_data(start_date, end_date, selected_assets, selected_regimes, n):
    # Create a date range from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    merged = pd.DataFrame({'DateTime': date_range})
    merged.set_index('DateTime', inplace=True)

    # For each asset, get its data and merge
    for asset in selected_assets:
        asset_data = asset_ts_data[['DateTime', asset]].copy()
        asset_data = asset_data[(asset_data['DateTime'] >= start_date) & (asset_data['DateTime'] <= end_date)]
        asset_data.set_index('DateTime', inplace=True)
        # Resample asset data to daily frequency without forward fill
        asset_data = asset_data.resample('D').mean()
        merged = merged.join(asset_data, how='left')

    # Get regime data
    regime_filtered = regime_data[['DateTime', f'sma_{n}_regime']].copy()
    regime_filtered = regime_filtered[(regime_filtered['DateTime'] >= start_date) & (regime_filtered['DateTime'] <= end_date)]
    regime_filtered.set_index('DateTime', inplace=True)
    # Resample regime data to daily frequency without forward fill
    regime_filtered = regime_filtered.resample('D').mean()

    # Merge regime data
    merged = merged.join(regime_filtered, how='left')

    # Map regime colors and labels
    merged['Regime Color'] = merged[f'sma_{n}_regime'].map(
        lambda x: regime_colors.get(int(x) if pd.notnull(x) else x, ('grey', 'Unknown'))[0]
    )
    merged['Regime Label'] = merged[f'sma_{n}_regime'].map(
        lambda x: regime_colors.get(int(x) if pd.notnull(x) else x, ('grey', 'Unknown'))[1]
    )

    merged.reset_index(inplace=True)
    return merged

with st.spinner('Processing data...'):
    merged_data = get_filtered_data(start_date, end_date, selected_assets, selected_regimes, n)

# Tabs for different analyses
tabs = st.tabs(["Regime Visualization", "Asset Performance Over Time", "Performance Metrics per Regime"])





















# Tab 1: Regime Visualization
with tabs[0]:
    st.subheader("Regime Visualization")
    
    # Checkboxes to show/hide curves
    show_sp500_sma = st.checkbox(f"Show S&P 500 SMA ({n}m)", value=True, key='regime_sp500_sma')
    show_inflation_rate_sma = st.checkbox(f"Show Inflation Rate SMA ({n}m)", value=True, key='regime_inflation_sma')
    show_sp500 = st.checkbox("Show S&P 500", value=False, key='regime_sp500')
    show_inflation_rate = st.checkbox("Show Inflation Rate", value=False, key='regime_inflation')
    
    # Checkboxes to toggle log scales
    log_scale_sp500 = st.checkbox("Log Scale for S&P 500 Axis", value=False, key='regime_log_sp500')
    log_scale_inflation_rate = st.checkbox("Log Scale for Inflation Rate Axis", value=False, key='regime_log_inflation')
    
    # Filter regime data for plotting
    regime_plot_data = regime_data[(regime_data['DateTime'] >= start_date) & (regime_data['DateTime'] <= end_date)].copy()
    
    # Map regime colors and labels
    regime_plot_data['Regime Color'] = regime_plot_data[f'sma_{n}_regime'].map(lambda x: regime_colors.get(x, ('grey', 'Unknown'))[0])
    regime_plot_data['Regime Label'] = regime_plot_data[f'sma_{n}_regime'].map(lambda x: regime_colors.get(x, ('grey', 'Unknown'))[1])
    
    # Identify regime change points and add shaded regions
    regime_plot_data['Regime Change'] = (regime_plot_data[f'sma_{n}_regime'] != regime_plot_data[f'sma_{n}_regime'].shift()).cumsum()
    regime_periods = regime_plot_data.groupby('Regime Change')
    
    # Initialize the plot
    fig = go.Figure()
    
    for name, group in regime_periods:
        regime_num = group[f'sma_{n}_regime'].iloc[0]
        color, label = regime_colors.get(regime_num, ('grey', 'Unknown'))
        start_date_regime = group['DateTime'].iloc[0]
        end_date_regime = group['DateTime'].iloc[-1]
        
        if start_date_regime == end_date_regime:
            end_date_regime += pd.Timedelta(days=1)
        
        fig.add_vrect(
            x0=start_date_regime,
            x1=end_date_regime,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0
        )
    
    # Flag to ensure Date and Regime are added once in the hover
    hover_header_added = False
    
    # Add traces based on user selection
    if show_sp500_sma:
        fig.add_trace(go.Scatter(
            x=regime_plot_data['DateTime'],
            y=regime_plot_data[f'sp500_sma_{n}'],
            mode='lines',
            name=f'S&P 500 SMA ({n}m)',
            line=dict(color='blue'),
            yaxis='y1',
            customdata=regime_plot_data['Regime Label'],
            hovertemplate=(
                'Date: %{x|%Y-%m-%d}<br>' +
                'Regime: %{customdata}<br>' +
                '%{fullData.name}: %{y:.2f}<extra></extra>'
            ),
            showlegend=False
        ))
        hover_header_added = True
    
    if show_inflation_rate_sma:
        if not hover_header_added:
            hover_template = (
                'Date: %{x|%Y-%m-%d}<br>' +
                'Regime: %{customdata}<br>' +
                '%{fullData.name}: %{y:.2f}<extra></extra>'
            )
            hover_header_added = True
        else:
            hover_template = '%{fullData.name}: %{y:.2f}<extra></extra>'
        fig.add_trace(go.Scatter(
            x=regime_plot_data['DateTime'],
            y=regime_plot_data[f'inflation_rate_sma_{n}'],
            mode='lines',
            name=f'Inflation Rate SMA ({n}m)',
            line=dict(color='red'),
            yaxis='y2',
            customdata=regime_plot_data['Regime Label'],
            hovertemplate=hover_template,
            showlegend=False
        ))
    
    if show_sp500:
        if not hover_header_added:
            hover_template = (
                'Date: %{x|%Y-%m-%d}<br>' +
                'Regime: %{customdata}<br>' +
                '%{fullData.name}: %{y:.2f}<extra></extra>'
            )
            hover_header_added = True
        else:
            hover_template = '%{fullData.name}: %{y:.2f}<extra></extra>'
        fig.add_trace(go.Scatter(
            x=regime_plot_data['DateTime'],
            y=regime_plot_data['S&P 500'],
            mode='lines',
            name='S&P 500',
            line=dict(color='blue', dash='dot'),
            yaxis='y1',
            customdata=regime_plot_data['Regime Label'],
            hovertemplate=hover_template,
            showlegend=False
        ))
    
    if show_inflation_rate:
        if not hover_header_added:
            hover_template = (
                'Date: %{x|%Y-%m-%d}<br>' +
                'Regime: %{customdata}<br>' +
                '%{fullData.name}: %{y:.2f}<extra></extra>'
            )
            hover_header_added = True
        else:
            hover_template = '%{fullData.name}: %{y:.2f}<extra></extra>'
        fig.add_trace(go.Scatter(
            x=regime_plot_data['DateTime'],
            y=regime_plot_data['Inflation Rate'],
            mode='lines',
            name='Inflation Rate',
            line=dict(color='red', dash='dot'),
            yaxis='y2',
            customdata=regime_plot_data['Regime Label'],
            hovertemplate=hover_template,
            showlegend=False
        ))
    
    # Update layout with optional log scales
    fig.update_layout(
        title=f'{n}-Month SMA of S&P 500 and Inflation Rate with {n}-Month SMA-Based Regimes',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='S&P 500',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            side='left',
            type='log' if log_scale_sp500 else 'linear'
        ),
        yaxis2=dict(
            title='Inflation Rate',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
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
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=False)
    
    
    # Create Curve Legend under the graph
    st.markdown("### Curve Legend")
    curve_legend_html = "<ul style='list-style-type:none;'>"
    
    if show_sp500_sma:
        curve_legend_html += f"<li><span style='color:blue;'>■</span> S&P 500 SMA ({n}m)</li>"
    if show_inflation_rate_sma:
        curve_legend_html += f"<li><span style='color:red;'>■</span> Inflation Rate SMA ({n}m)</li>"
    if show_sp500:
        curve_legend_html += f"<li><span style='border-bottom: 2px dashed blue; display:inline-block; width:15px; margin-right:5px;'></span> S&P 500</li>"
    if show_inflation_rate:
        curve_legend_html += f"<li><span style='border-bottom: 2px dashed red; display:inline-block; width:15px; margin-right:5px;'></span> Inflation Rate</li>"
    
    curve_legend_html += "</ul>"
    st.markdown(curve_legend_html, unsafe_allow_html=True)
    
    # Create Regime Legend under the graph
    st.markdown("### Regime Legend")
    regime_legend_html = "<ul style='list-style-type:none;'>"
    for regime_num, (color, label) in regime_colors.items():
        regime_legend_html += f"<li><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px;'></span> {label}</li>"
    # Add legend for Unknown regime
    regime_legend_html += f"<li><span style='background-color:grey; width:15px; height:15px; display:inline-block; margin-right:5px;'></span> Unknown (NaN)</li>"
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
    csv = regime_plot_data.to_csv(index=False)
    st.download_button(
        label="Download Regime Data as CSV",
        data=csv,
        file_name='regime_data.csv',
        mime='text/csv',
    )






















# Tab 2: Asset Performance Over Time
with tabs[1]:
    st.subheader("Asset Performance Over Time")

    # Add checkbox for log scale
    log_scale_normalized = st.checkbox(
        "Log Scale for Normalized Prices", value=False, key='log_scale_normalized'
    )

    # Check if assets are selected
    if not selected_assets:
        st.warning("Please select at least one asset to display.")
    else:
        # Initialize the plot
        fig2 = go.Figure()

        # For regime shading, use the regime data without resampling
        regime_plot_data = regime_data[['DateTime', f'sma_{n}_regime']].copy()
        regime_plot_data = regime_plot_data[
            (regime_plot_data['DateTime'] >= start_date) & (regime_plot_data['DateTime'] <= end_date)
        ]
        regime_plot_data.sort_values('DateTime', inplace=True)
        regime_plot_data.reset_index(drop=True, inplace=True)

        # Map regime colors and labels
        regime_plot_data['Regime Color'] = regime_plot_data[f'sma_{n}_regime'].map(
            lambda x: regime_colors.get(int(x) if pd.notnull(x) else x, ('grey', 'Unknown'))[0]
        )
        regime_plot_data['Regime Label'] = regime_plot_data[f'sma_{n}_regime'].map(
            lambda x: regime_colors.get(int(x) if pd.notnull(x) else x, ('grey', 'Unknown'))[1]
        )

        # Identify regime change points and add shaded regions
        regime_plot_data['Regime Change'] = (
            regime_plot_data[f'sma_{n}_regime'] != regime_plot_data[f'sma_{n}_regime'].shift()
        ).cumsum()
        regime_periods = regime_plot_data.groupby('Regime Change')

        for name, group in regime_periods:
            regime_num = group[f'sma_{n}_regime'].iloc[0]
            if pd.isnull(regime_num):
                continue  # Skip if regime_num is NaN
            if regime_num not in selected_regimes:
                continue  # Skip regimes not selected
            color, label = regime_colors.get(int(regime_num), ('grey', 'Unknown'))
            start_date_regime = group['DateTime'].iloc[0]
            end_date_regime = group['DateTime'].iloc[-1]

            if start_date_regime == end_date_regime:
                end_date_regime += pd.Timedelta(days=1)

            fig2.add_vrect(
                x0=start_date_regime,
                x1=end_date_regime,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0
            )

        # Add asset traces without resampling or forward-filling
        for asset in selected_assets:
            asset_data = asset_ts_data[['DateTime', asset]].copy()
            asset_data = asset_data[
                (asset_data['DateTime'] >= start_date) & (asset_data['DateTime'] <= end_date)
            ]
            asset_data.dropna(subset=[asset], inplace=True)  # Remove NaN values

            if asset_data.empty:
                st.warning(f"No data available for asset {asset} in the selected date range.")
                continue

            # Store actual prices
            asset_data['Actual Price'] = asset_data[asset]

            # Normalize prices so that the last valid point is 100
            last_valid_value = asset_data[asset].iloc[-1]
            asset_data[asset] = (asset_data[asset] / last_valid_value) * 100

            # Prepare customdata with actual prices
            customdata = asset_data['Actual Price'].values

            fig2.add_trace(go.Scatter(
                x=asset_data['DateTime'],
                y=asset_data[asset],
                mode='lines',
                name=asset,
                customdata=customdata,
                connectgaps=False,  # Do not connect gaps
                hovertemplate=(
                    f"{asset}<br>" +
                    "Date: %{x|%Y-%m-%d}<br>" +
                    "Normalized Price: %{y:.2f}<br>" +
                    "Actual Price: %{customdata:.2f}<extra></extra>"
                )
            ))

        # Update layout
        fig2.update_layout(
            title='Asset Performance Over Time (Normalized to 100 at Last Available Date)',
            xaxis=dict(title='Date', range=[start_date, end_date]),
            yaxis=dict(
                title='Normalized Price',
                type='log' if log_scale_normalized else 'linear'
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

        # Display the plot
        st.plotly_chart(fig2, use_container_width=False)

        # Export plot as image
        buffer = io.BytesIO()
        fig2.write_image(buffer, format='png')
        st.download_button(
            label="Download Plot as PNG",
            data=buffer,
            file_name='asset_performance_plot.png',
            mime='image/png'
        )

        # Provide a download button for the asset data
        # Merge all selected asset data for download
        all_asset_data = pd.DataFrame()
        for asset in selected_assets:
            asset_data = asset_ts_data[['DateTime', asset]].copy()
            asset_data = asset_data[
                (asset_data['DateTime'] >= start_date) & (asset_data['DateTime'] <= end_date)
            ]
            if all_asset_data.empty:
                all_asset_data = asset_data
            else:
                all_asset_data = pd.merge(
                    all_asset_data, asset_data, on='DateTime', how='outer'
                )

        # Sort the combined data by DateTime
        all_asset_data.sort_values('DateTime', inplace=True)
        csv = all_asset_data.to_csv(index=False)
        st.download_button(
            label="Download Asset Data as CSV",
            data=csv,
            file_name='asset_data.csv',
            mime='text/csv',
        )
























# Tab 3: Performance Metrics per Regime
with tabs[2]:
    st.subheader("Performance Metrics per Regime")
    
    # Check if assets and metrics are selected
    if not selected_assets or not selected_metrics:
        st.warning("Please select at least one asset and one performance metric to display.")
    else:
        # Ensure 'DateTime' is datetime type in merged_data
        merged_data['DateTime'] = pd.to_datetime(merged_data['DateTime'])
        
        # For each asset and each regime, compute the performance metrics
        performance_results = []
        
        for asset in selected_assets:
            # For the asset, get the data
            asset_data = merged_data[['DateTime', asset, f'sma_{n}_regime']].copy()
            
            # Drop rows with NaN in asset or regime
            asset_data.dropna(subset=[asset, f'sma_{n}_regime'], inplace=True)
            
            # For each regime in selected_regimes
            for regime in selected_regimes:
                # Filter data for the regime
                regime_data = asset_data[asset_data[f'sma_{n}_regime'] == regime]
                
                # Check if we have enough data
                if len(regime_data) > 1:
                    # Compute daily returns
                    regime_data['Return'] = regime_data[asset].pct_change()
                    
                    # Compute performance metrics
                    avg_return = regime_data['Return'].mean() * 252  # annualized average return
                    volatility = regime_data['Return'].std() * np.sqrt(252)  # annualized volatility
                    sharpe_ratio = avg_return / volatility if volatility != 0 else np.nan
                    # Compute Max Drawdown
                    cumulative = (1 + regime_data['Return'].fillna(0)).cumprod()
                    cumulative_max = cumulative.cummax()
                    drawdown = cumulative / cumulative_max - 1
                    max_drawdown = drawdown.min()
                    
                    # Append to results
                    performance_results.append({
                        'Asset': asset,
                        'Regime': regime,
                        'Average Return': avg_return,
                        'Volatility': volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Max Drawdown': max_drawdown
                    })
                else:
                    # Not enough data to compute metrics
                    performance_results.append({
                        'Asset': asset,
                        'Regime': regime,
                        'Average Return': np.nan,
                        'Volatility': np.nan,
                        'Sharpe Ratio': np.nan,
                        'Max Drawdown': np.nan
                    })
        
        # Convert to DataFrame
        perf_data_filtered = pd.DataFrame(performance_results)
        
        # Check if data is available
        if perf_data_filtered.empty:
            st.warning("No performance data available for the selected options.")
        else:
            # Pivot the data for better display
            pivot_table = perf_data_filtered.pivot(index='Asset', columns='Regime')[selected_metrics]
            
            # Display the table
            st.dataframe(pivot_table)
            
            # Bar Charts for each metric
            for metric in selected_metrics:
                st.markdown(f"#### {metric} by Asset and Regime")
                fig3 = go.Figure()
                
                for asset in selected_assets:
                    asset_data = perf_data_filtered[perf_data_filtered['Asset'] == asset]
                    fig3.add_trace(go.Bar(
                        x=asset_data['Regime'].astype(str),
                        y=asset_data[metric],
                        name=asset
                    ))
                
                # Update layout
                fig3.update_layout(
                    barmode='group',
                    xaxis=dict(title='Regime'),
                    yaxis=dict(title=metric),
                    title=f'{metric} by Asset and Regime',
                    width=800,
                    height=500
                )
                
                # Display the plot
                st.plotly_chart(fig3, use_container_width=False)
            
            # Provide a download button for the performance data
            csv = perf_data_filtered.to_csv(index=False)
            st.download_button(
                label="Download Performance Metrics Data as CSV",
                data=csv,
                file_name='performance_metrics.csv',
                mime='text/csv',
            )
