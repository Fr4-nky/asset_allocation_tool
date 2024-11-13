# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io  # For exporting plot as image

# Import kaleido for saving plot as image
import plotly.io as pio

# Set page configuration
st.set_page_config(
    page_title="S&P 500 and Inflation Rate Regimes",
    layout="wide",
    page_icon="📈"
)

st.title("Macroeconomic Regimes Visualization")
st.write("""
This app visualizes the macroeconomic regimes based on S&P 500 and Inflation Rate data.
Select the moving average period and explore how regimes change over time.
""")

st.sidebar.header("User Input Parameters")

# List of 'n' values in months
n_values = [int(i * 12) for i in [1/6, 1/4, 1/2, 3/4, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15]]

# Use select_slider for non-uniform steps
n = st.sidebar.select_slider("Select n (months) for Moving Average:", options=n_values, value=12)

# Load the precomputed data
@st.cache_data
def load_data():
    path = './processed_data/'
    df = pd.read_csv(path + 'sp500_and_inflation_precomputed.csv', parse_dates=['DateTime'])
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.copy()

with st.spinner('Loading data...'):
    data = load_data()

# Ensure 'DateTime' is datetime type
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Data range selection
min_date = data['DateTime'].min()
max_date = data['DateTime'].max()

start_date = st.sidebar.date_input('Start date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End date', max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date.')
    st.stop()

# Filter data based on date range
mask = (data['DateTime'] >= pd.to_datetime(start_date)) & (data['DateTime'] <= pd.to_datetime(end_date))
data = data.loc[mask]

# Checkboxes to show/hide curves
show_sp500_sma = st.sidebar.checkbox(f"Show S&P 500 SMA ({n}m)", value=True)
show_inflation_rate_sma = st.sidebar.checkbox(f"Show Inflation Rate SMA ({n}m)", value=True)
show_sp500 = st.sidebar.checkbox("Show S&P 500", value=False)
show_inflation_rate = st.sidebar.checkbox("Show Inflation Rate", value=False)

# Checkboxes to toggle log scales
log_scale_sp500 = st.sidebar.checkbox("Log Scale for S&P 500 Axis", value=False)
log_scale_inflation_rate = st.sidebar.checkbox("Log Scale for Inflation Rate Axis", value=False)

# Select relevant columns based on n
sp500_sma_col = f'sp500_sma_{n}'
inflation_rate_sma_col = f'inflation_rate_sma_{n}'
sp500_sma_derivative_col = f'sp500_sma_{n}_derivative'
inflation_rate_sma_derivative_col = f'inflation_rate_sma_{n}_derivative'
sma_regime_col = f'sma_{n}_regime'

# Ensure that the required columns exist
required_columns = [
    sp500_sma_col,
    inflation_rate_sma_col,
    sp500_sma_derivative_col,
    inflation_rate_sma_derivative_col,
    sma_regime_col,
    'S&P 500',
    'Inflation Rate'
]

for col in required_columns:
    if col not in data.columns:
        st.error(f"Column {col} not found in data.")
        st.stop()

# Map regime numbers to colors and labels for visualization
regime_colors = {
    1: ('green', 'Rising S&P 500 & Rising Inflation Rate'),
    2: ('yellow', 'Rising S&P 500 & Falling Inflation Rate'),
    3: ('orange', 'Falling S&P 500 & Rising Inflation Rate'),
    4: ('red', 'Falling S&P 500 & Falling Inflation Rate')
}

data['Regime Color'] = data[sma_regime_col].map(lambda x: regime_colors[x][0])
data['Regime Label'] = data[sma_regime_col].map(lambda x: regime_colors[x][1])

# Identify regime change points and add shaded regions
data['Regime Change'] = (data[sma_regime_col] != data[sma_regime_col].shift()).cumsum()
regime_periods = data.groupby('Regime Change')

# Initialize the plot
fig = go.Figure()

for name, group in regime_periods:
    regime_num = group[sma_regime_col].iloc[0]
    color, label = regime_colors[regime_num]
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
        x=data['DateTime'],
        y=data[sp500_sma_col],
        mode='lines',
        name=f'S&P 500 SMA ({n}m)',
        line=dict(color='blue'),
        yaxis='y1',
        customdata=data['Regime Label'],
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
        x=data['DateTime'],
        y=data[inflation_rate_sma_col],
        mode='lines',
        name=f'Inflation Rate SMA ({n}m)',
        line=dict(color='red'),
        yaxis='y2',
        customdata=data['Regime Label'],
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
        x=data['DateTime'],
        y=data['S&P 500'],
        mode='lines',
        name='S&P 500',
        line=dict(color='blue', dash='dot'),
        yaxis='y1',
        customdata=data['Regime Label'],
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
        x=data['DateTime'],
        y=data['Inflation Rate'],
        mode='lines',
        name='Inflation Rate',
        line=dict(color='red', dash='dot'),
        yaxis='y2',
        customdata=data['Regime Label'],
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

# Export plot as image
buffer = io.BytesIO()
fig.write_image(buffer, format='png')
st.sidebar.download_button(
    label="Download Plot as PNG",
    data=buffer,
    file_name='plot.png',
    mime='image/png'
)

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
regime_legend_html += "</ul>"
st.markdown(regime_legend_html, unsafe_allow_html=True)

# Provide a download button for the data
csv = data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Data with Regimes as CSV",
    data=csv,
    file_name='data_with_regimes.csv',
    mime='text/csv',
)

# Statistical Summary Table
if st.sidebar.checkbox("Show Statistical Summary Table", value=False):
    st.markdown("### Statistical Summary Table")
    stats_cols = [
        'S&P 500', sp500_sma_col, sp500_sma_derivative_col,
        'Inflation Rate', inflation_rate_sma_col, inflation_rate_sma_derivative_col
    ]
    stats_df = data[stats_cols].describe().T
    st.dataframe(stats_df)

