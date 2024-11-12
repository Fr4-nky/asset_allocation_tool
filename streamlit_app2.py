import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configure the Streamlit app to use the full width of the page
st.set_page_config(layout="wide")

st.title("Macroeconomic Regimes Visualization")
st.write("""
This app visualizes the macroeconomic regimes based on Growth and Inflation data.
Select the moving average period and explore how regimes change over time.
""")

st.sidebar.header("User Input Parameters")

# List of 'n' values in months
n_values = [int(i * 12) for i in [1/6, 1/4, 1/2, 3/4, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15]]

# Use select_slider for non-uniform steps
n = st.sidebar.select_slider("Select n (months) for Moving Average:", options=n_values, value=12)

# Checkboxes to show/hide curves
show_growth_sma = st.sidebar.checkbox(f"Show Growth SMA ({n}m)", value=True)
show_inflation_sma = st.sidebar.checkbox(f"Show Inflation SMA ({n}m)", value=True)
show_growth = st.sidebar.checkbox("Show Growth", value=False)
show_inflation = st.sidebar.checkbox("Show Inflation", value=False)

# Checkboxes to toggle log scales
log_scale_growth = st.sidebar.checkbox("Log Scale for Growth Axis", value=False)
log_scale_inflation = st.sidebar.checkbox("Log Scale for Inflation Axis", value=False)

# Load the data
path = './processed_data/' 

@st.cache_data
def load_data():
    df = pd.read_csv(path + 'growth_and_inflation2.csv', parse_dates=['DateTime'])
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.copy()

data = load_data()

# Ensure 'DateTime' is datetime type
data['DateTime'] = pd.to_datetime(data['DateTime'])

def compute_sma(df, n):
    df[f'growth_sma_{n}'] = df['Growth'].rolling(window=n).mean()
    df[f'inflation_sma_{n}'] = df['Inflation'].rolling(window=n).mean()
    return df

def compute_derivatives(df, n):
    df[f'growth_sma_{n}_derivative'] = df[f'growth_sma_{n}'].diff()
    df[f'inflation_sma_{n}_derivative'] = df[f'inflation_sma_{n}'].diff()
    return df

def determine_regimes(df, n):
    def regime(row):
        if row[f'growth_sma_{n}_derivative'] > 0 and row[f'inflation_sma_{n}_derivative'] > 0:
            return 1  # Rising Growth & Rising Inflation
        elif row[f'growth_sma_{n}_derivative'] > 0 and row[f'inflation_sma_{n}_derivative'] <= 0:
            return 2  # Rising Growth & Falling Inflation
        elif row[f'growth_sma_{n}_derivative'] <= 0 and row[f'inflation_sma_{n}_derivative'] > 0:
            return 3  # Falling Growth & Rising Inflation
        else:
            return 4  # Falling Growth & Falling Inflation

    df[f'sma_{n}_regime'] = df.apply(regime, axis=1)
    return df

regime_colors = {
    1: ('green', 'Rising Growth & Rising Inflation'),
    2: ('yellow', 'Rising Growth & Falling Inflation'),
    3: ('orange', 'Falling Growth & Rising Inflation'),
    4: ('red', 'Falling Growth & Falling Inflation')
}

# Apply computations
data = compute_sma(data, n)
data = compute_derivatives(data, n)
data = determine_regimes(data, n)

# Save the DataFrame to CSV and provide a download button
csv = data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Data with Regimes as CSV",
    data=csv,
    file_name='data_with_regimes.csv',
    mime='text/csv',
)
st.write("You can download the processed data as a CSV file from the sidebar.")

# Map regime numbers to colors and labels for visualization
data['Regime Color'] = data[f'sma_{n}_regime'].map(lambda x: regime_colors[x][0])
data['Regime Label'] = data[f'sma_{n}_regime'].map(lambda x: regime_colors[x][1])

# Initialize the plot
fig = go.Figure()

# Identify regime change points and add shaded regions
data['Regime Change'] = (data[f'sma_{n}_regime'] != data[f'sma_{n}_regime'].shift()).cumsum()
regime_periods = data.groupby('Regime Change')

for name, group in regime_periods:
    regime_num = group[f'sma_{n}_regime'].iloc[0]
    color, label = regime_colors[regime_num]
    start_date = group['DateTime'].iloc[0]
    end_date = group['DateTime'].iloc[-1]

    if start_date == end_date:
        # Extend end_date by a small amount to ensure visibility
        end_date += pd.Timedelta(days=1)

    fig.add_vrect(
        x0=start_date,
        x1=end_date,
        fillcolor=color,
        opacity=0.3,
        layer="below",
        line_width=0
    )

# Flag to ensure Date and Regime are added once in the hover
hover_header_added = False

# Add traces based on user selection
if show_growth_sma:
    fig.add_trace(go.Scatter(
        x=data['DateTime'],
        y=data[f'growth_sma_{n}'],
        mode='lines',
        name=f'Growth SMA ({n}m)',
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

if show_inflation_sma:
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
        y=data[f'inflation_sma_{n}'],
        mode='lines',
        name=f'Inflation SMA ({n}m)',
        line=dict(color='red'),
        yaxis='y2',
        customdata=data['Regime Label'],
        hovertemplate=hover_template,
        showlegend=False
    ))

if show_growth:
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
        y=data['Growth'],
        mode='lines',
        name='Growth',
        line=dict(color='blue', dash='dot'),
        yaxis='y1',
        customdata=data['Regime Label'],
        hovertemplate=hover_template,
        showlegend=False
    ))

if show_inflation:
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
        y=data['Inflation'],
        mode='lines',
        name='Inflation',
        line=dict(color='red', dash='dot'),
        yaxis='y2',
        customdata=data['Regime Label'],
        hovertemplate=hover_template,
        showlegend=False
    ))

# Update layout with optional log scales
fig.update_layout(
    title=f'{n}-Month SMA of Growth and Inflation with {n}-Month SMA-Based Regimes',
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='Growth',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        side='left',
        type='log' if log_scale_growth else 'linear'
    ),
    yaxis2=dict(
        title='Inflation',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right',
        type='log' if log_scale_inflation else 'linear'
    ),
    hovermode='x unified',
    width=1200,  # Increased figure width
    height=700,  # Increased figure height
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

if show_growth_sma:
    # Solid square for Growth SMA
    curve_legend_html += f"<li><span style='color:blue;'>■</span> Growth SMA ({n}m)</li>"
if show_inflation_sma:
    # Solid square for Inflation SMA
    curve_legend_html += f"<li><span style='color:red;'>■</span> Inflation SMA ({n}m)</li>"
if show_growth:
    # Dashed line for Growth
    curve_legend_html += f"<li><span style='border-bottom: 2px dashed blue; display:inline-block; width:15px; margin-right:5px;'></span> Growth</li>"
if show_inflation:
    # Dashed line for Inflation
    curve_legend_html += f"<li><span style='border-bottom: 2px dashed red; display:inline-block; width:15px; margin-right:5px;'></span> Inflation</li>"

curve_legend_html += "</ul>"
st.markdown(curve_legend_html, unsafe_allow_html=True)

# Create Regime Legend under the graph
st.markdown("### Regime Legend")
regime_legend_html = "<ul style='list-style-type:none;'>"
for regime_num, (color, label) in regime_colors.items():
    regime_legend_html += f"<li><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px;'></span> {label}</li>"
regime_legend_html += "</ul>"
st.markdown(regime_legend_html, unsafe_allow_html=True)
