import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core.constants import asset_colors, regime_bg_colors, regime_labels_dict, regime_legend_colors

def render(tab, sp_inflation_data):
    """
    Render the Regime Visualization tab: macro regime timeline with S&P500 and Inflation.
    """
    # Retrieve MA window from sidebar state
    sp500_n = st.session_state['ma_length']
    inflation_n = sp500_n

    tab.markdown(
        """
        <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Visualization</h2>
        """, unsafe_allow_html=True
    )
    t_tab1 = time.time()
    print("DEBUG: Rendering Tab 1: Regime Visualization.")

    # Settings for chart (all enabled)
    show_sp500_ma = True
    show_inflation_ma = True
    show_sp500 = True
    show_inflation = True
    log_scale_sp500 = False
    log_scale_inflation_rate = False

    # Initialize figure
    fig = go.Figure()
    print("DEBUG: Tab 1 - Initialized go.Figure.")

    # Compute continuous regime segments
    sp_inflation_data['Regime_Change'] = (
        sp_inflation_data['Regime'] != sp_inflation_data['Regime'].shift()
    ).cumsum()
    grouped = sp_inflation_data.groupby(['Regime', 'Regime_Change'])
    print(f"DEBUG: Tab 1 - Grouped regimes. Groups: {len(grouped)}")

    # Collect periods
    periods = []
    for (regime, _), grp in grouped:
        periods.append({
            'Regime': regime,
            'Start': grp['DateTime'].iloc[0],
            'End': grp['DateTime'].iloc[-1]
        })

    # Sort and adjust
    dfp = pd.DataFrame(periods).sort_values('Start').reset_index(drop=True)
    max_date = sp_inflation_data['DateTime'].max()
    for i in dfp.index:
        start = dfp.loc[i, 'Start']
        regime = dfp.loc[i, 'Regime']
        end = dfp.loc[i+1, 'Start'] if i < len(dfp)-1 else max_date
        if end < start:
            end = start
        color = regime_bg_colors.get(regime, 'rgba(200,200,200,0.10)')
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=1.0, layer="below", line_width=0)
    print(f"DEBUG: Tab 1 - Added {len(dfp)} vrects.")

    # Prepare customdata for hover
    req = ['Regime', 'S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']
    # Map regimes to labels, fill NaNs
    labels = sp_inflation_data['Regime'].map(lambda x: regime_labels_dict.get(x, 'Unknown'))
    data = [sp_inflation_data['S&P 500'].fillna(0),
            sp_inflation_data['S&P 500 MA'].fillna(0),
            sp_inflation_data['Inflation Rate'].fillna(0),
            sp_inflation_data['Inflation Rate MA'].fillna(0),
            labels]
    customdata = np.stack(data, axis=-1)
    print(f"DEBUG: Tab 1 - Customdata shape: {customdata.shape}")

    # Add traces
    if show_sp500:
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500'],
            mode='lines',
            name='S&P 500',
            line=dict(color=asset_colors['S&P 500'], dash='dot'),
            yaxis='y1',
            customdata=customdata,
            hovertemplate='S&P 500: %{customdata[0]:.2f}<br>Regime: %{customdata[4]}<extra></extra>'
        ))
    if show_sp500_ma:
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['S&P 500 MA'],
            mode='lines',
            name=f'S&P 500 MA ({sp500_n}m)',
            line=dict(color=asset_colors['S&P 500 MA']),
            yaxis='y1',
            customdata=customdata,
            hovertemplate='S&P 500 MA: %{customdata[1]:.2f}<br>Regime: %{customdata[4]}<extra></extra>'
        ))
    if show_inflation:
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate'],
            mode='lines',
            name='Inflation Rate',
            line=dict(color=asset_colors['Inflation Rate'], dash='dot'),
            yaxis='y2',
            customdata=customdata,
            hovertemplate='Inflation Rate: %{customdata[2]:.2f}<br>Regime: %{customdata[4]}<extra></extra>'
        ))
    if show_inflation_ma:
        fig.add_trace(go.Scatter(
            x=sp_inflation_data['DateTime'],
            y=sp_inflation_data['Inflation Rate MA'],
            mode='lines',
            name=f'Inflation Rate MA ({inflation_n}m)',
            line=dict(color=asset_colors['Inflation Rate MA']),
            yaxis='y2',
            customdata=customdata,
            hovertemplate='Inflation Rate MA: %{customdata[3]:.2f}<br>Regime: %{customdata[4]}<extra></extra>'
        ))
    print("DEBUG: Tab 1 - Added all traces.")

    # Layout
    fig.update_layout(
        title={'text': 'Macro Regime Timeline: S&P 500 & Inflation', 'x': 0.5},
        xaxis=dict(title='Date'),
        yaxis=dict(title='S&P 500', type='log' if log_scale_sp500 else 'linear'),
        yaxis2=dict(title='Inflation Rate', overlaying='y', side='right', type='log' if log_scale_inflation_rate else 'linear'),
        hovermode='x unified', width=1200, height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=False)

    # Regime Legend with transparent color swatches (dynamic)
    custom_legend_names = {
        2: "👧🏼 <b>Goldilocks</b>: Rising growth, falling inflation",
        1: "🎈 <b>Reflation</b>: Rising growth, rising inflation",
        4: "💨 <b>Deflation</b>: Falling growth, falling inflation",
        3: "✋ <b>Stagflation</b>: Falling growth, rising inflation"
    }
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Legend</h2>
    <ul style='list-style-type:none; padding-left:0;'>
    """ + "\n".join([
        f"<li><span style='background-color:{regime_legend_colors.get(regime_num, 'grey')}; width:15px; height:15px; display:inline-block; margin-right:5px; border-radius:3px; border:1px solid #888;'></span> {custom_legend_names[regime_num]}</li>"
        for regime_num in [2, 1, 4, 3]
    ]) + "\n</ul>", unsafe_allow_html=True)

    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Scatter Plots</h2>
    """, unsafe_allow_html=True)

    # --- Regime Scatter Plots ---
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    # Prepare data for plotting
    derivative_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime']].dropna().copy()
    derivative_df['S&P 500 MA Pct Change'] = sp_inflation_data['S&P 500 MA'].pct_change()
    derivative_df = derivative_df.sort_values('DateTime').reset_index(drop=True)
    window_size = 54
    window_df = derivative_df.iloc[-window_size:].copy()
    for col in ['S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']:
        if col not in window_df.columns:
            window_df[col] = np.nan
    if not window_df.empty:
        date_start = window_df['DateTime'].iloc[0].strftime('%Y-%m-%d')
        date_end = window_df['DateTime'].iloc[-1].strftime('%Y-%m-%d')
        date_span = f"{date_start} to {date_end}"
    else:
        date_span = "N/A"
    by_cmap = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=window_size)
    colors = [mcolors.to_hex(by_cmap(i/(window_size-1))) for i in range(len(window_df))]
    scatter_fig = go.Figure()
    x_min = float(window_df['S&P 500 MA Pct Change'].min()) if not window_df.empty else -1
    x_max = float(window_df['S&P 500 MA Pct Change'].max()) if not window_df.empty else 1
    y_min = float(window_df['Inflation Rate MA Growth'].min()) if not window_df.empty else -1
    y_max = float(window_df['Inflation Rate MA Growth'].max()) if not window_df.empty else 1
    x_margin = 0.1 * (x_max - x_min) if x_max != x_min else 0.1
    y_margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.1
    x_bg_min = min(x_min, 0) - x_margin
    x_bg_max = max(x_max, 0) + x_margin
    y_bg_min = min(y_min, 0) - y_margin
    y_bg_max = max(y_max, 0) + y_margin
    x_range = [x_bg_min, x_bg_max]
    y_range = [y_bg_min, y_bg_max]
    scatter_fig.add_shape(type="rect", x0=0, x1=x_bg_max, y0=0, y1=y_bg_max, fillcolor=regime_bg_colors[1], line_width=0, layer="below")
    scatter_fig.add_shape(type="rect", x0=0, x1=x_bg_max, y0=y_bg_min, y1=0, fillcolor=regime_bg_colors[2], line_width=0, layer="below")
    scatter_fig.add_shape(type="rect", x0=x_bg_min, x1=0, y0=0, y1=y_bg_max, fillcolor=regime_bg_colors[3], line_width=0, layer="below")
    scatter_fig.add_shape(type="rect", x0=x_bg_min, x1=0, y0=y_bg_min, y1=0, fillcolor=regime_bg_colors[4], line_width=0, layer="below")
    scatter_fig.add_trace(go.Scatter(
        x=window_df['S&P 500 MA Pct Change'],
        y=window_df['Inflation Rate MA Growth'],
        mode='lines+markers',
        marker=dict(color=colors, size=12, line=dict(width=1, color='black')),
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

    # --- Second Scatter Plot (All Data) ---
    all_df = sp_inflation_data[['DateTime', 'S&P 500 MA Growth', 'Inflation Rate MA Growth', 'Regime', 'S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']].dropna().copy()
    all_df['S&P 500 MA Pct Change'] = sp_inflation_data['S&P 500 MA'].pct_change()
    for col in ['S&P 500', 'S&P 500 MA', 'Inflation Rate', 'Inflation Rate MA']:
        if col not in all_df.columns:
            all_df.loc[:, col] = np.nan
    N_all = len(all_df)
    by_cmap_all = LinearSegmentedColormap.from_list('by', ['black', 'yellow'], N=N_all)
    all_colors = [mcolors.to_hex(by_cmap_all(i/(N_all-1))) for i in range(N_all)]
    all_scatter_fig = go.Figure()
    x_min_all = float(all_df['S&P 500 MA Pct Change'].min()) if not all_df.empty else -1
    x_max_all = float(all_df['S&P 500 MA Pct Change'].max()) if not all_df.empty else 1
    y_min_all = float(all_df['Inflation Rate MA Growth'].min()) if not all_df.empty else -1
    y_max_all = float(all_df['Inflation Rate MA Growth'].max()) if not all_df.empty else 1
    x_margin_all = 0.1 * (x_max_all - x_min_all) if x_max_all != x_min_all else 0.1
    y_margin_all = 0.1 * (y_max_all - y_min_all) if y_max_all != y_min_all else 0.1
    x_bg_min_all = min(x_min_all, 0) - x_margin_all
    x_bg_max_all = max(x_max_all, 0) + x_margin_all
    y_bg_min_all = min(y_min_all, 0) - y_margin_all
    y_bg_max_all = max(y_max_all, 0) + y_margin_all
    x_range_all = [x_bg_min_all, x_bg_max_all]
    y_range_all = [y_bg_min_all, y_bg_max_all]
    all_scatter_fig.add_shape(type="rect", x0=0, x1=x_bg_max_all, y0=0, y1=y_bg_max_all, fillcolor=regime_bg_colors[1], line_width=0, layer="below")
    all_scatter_fig.add_shape(type="rect", x0=0, x1=x_bg_max_all, y0=y_bg_min_all, y1=0, fillcolor=regime_bg_colors[2], line_width=0, layer="below")
    all_scatter_fig.add_shape(type="rect", x0=x_bg_min_all, x1=0, y0=0, y1=y_bg_max_all, fillcolor=regime_bg_colors[3], line_width=0, layer="below")
    all_scatter_fig.add_shape(type="rect", x0=x_bg_min_all, x1=0, y0=y_bg_min_all, y1=0, fillcolor=regime_bg_colors[4], line_width=0, layer="below")
    all_scatter_fig.add_trace(go.Scatter(
        x=all_df['S&P 500 MA Pct Change'],
        y=all_df['Inflation Rate MA Growth'],
        mode='markers',
        marker=dict(color=all_colors, size=10, line=dict(width=1, color='black')),
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

    # --- Regime Periods Table ---
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Regime Periods</h2>
    """, unsafe_allow_html=True)
    dfp = sp_inflation_data.dropna(subset=['Regime'])
    dfp = dfp[dfp['Regime'] != 'Unknown']
    change_mask = dfp['Regime'].ne(dfp['Regime'].shift())
    df_start = dfp.loc[change_mask, ['DateTime', 'Regime']].copy()
    df_start['Start Date'] = df_start['DateTime'].dt.strftime('%Y-%m-%d')
    df_start['End Date'] = df_start['Start Date'].shift(-1)
    last_date = dfp['DateTime'].iloc[-1].strftime('%Y-%m-%d')
    df_start.at[df_start.index[-1], 'End Date'] = last_date
    df_start['Regime'] = df_start['Regime'].map(regime_labels_dict)
    periods_df = df_start[['Regime', 'Start Date', 'End Date']]
    periods_df = periods_df.sort_values('Start Date', ascending=False).reset_index(drop=True)
    def highlight_regime_period(row):
        lbl = row['Regime']
        num = next((k for k,v in regime_labels_dict.items() if v==lbl), None)
        css = regime_bg_colors.get(num, '#eeeeee')
        if css.startswith('rgba'):
            import re
            m = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css)
            if m:
                r,g,b,_ = m.groups()
                from core.constants import REGIME_BG_ALPHA
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

# <li>S&amp;P 500:
#     <ul>
#       <li>Recent Prices: <a href="https://fred.stlouisfed.org/series/SP500" target="_blank">FRED S&amp;P 500</a></li>
#       <li>From 1928 until 2023: <a href="https://finance.yahoo.com/quote/%5ESPX/history/" target="_blank">Yahoo Finance S&amp;P 500</a></li>
#       <li>Until 1927: <a href="https://www.multpl.com/s-p-500-historical-prices/table/by-month" target="_blank">Multpl S&amp;P 500</a></li>
#     </ul>
#   </li>
#   <li>Total Return Bond Index:
#     <ul>
#       <li>Since 1973: <a href="https://fred.stlouisfed.org/series/BAMLCC0A0CMTRIV" target="_blank">FRED BAMLCC0A0CMTRIV</a></li>
#       <li>Until 1973: <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3805927" target="_blank">McQuarrie: Where Siegel Went Awry: Outdated Sources &amp; Incomplete Data</a></li>
#     </ul>
#   </li>
#   <li>Gold:
#     <ul>
#       <li>Since 2000: <a href="https://finance.yahoo.com/quote/GC=F/" target="_blank">Yahoo Finance: COMEX Gold</a></li>
#       <li>Until 2000: <a href="https://onlygold.com/gold-prices/historical-gold-prices/" target="_blank">OnlyGold: Historical Prices</a></li>
#     </ul>
#   </li>
#   <li>Inflation:
#     <ul>
#       <li>Latest data point: <a href="https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting" target="_blank">Cleveland Fed: Inflation Nowcasting</a></li>
#       <li>Since 1913: <a href="https://fred.stlouisfed.org/series/CPIAUCNS" target="_blank">FRED: CPI</a></li>
#       <li>Until 1913: <a href="http://www.econ.yale.edu/~shiller/data.htm" target="_blank">Yale University: Online Data Robert Shiller</a></li>
#     </ul>
#   </li>

# https://finance.yahoo.com/quote/SPY/
# https://finance.yahoo.com/quote/VT/
# https://finance.yahoo.com/quote/TLT/
# https://finance.yahoo.com/quote/IEF/
# https://finance.yahoo.com/quote/TIP/
# https://finance.yahoo.com/quote/DBC/
# https://finance.yahoo.com/quote/GLD/


    # --- FURTHER INFORMATION SECTION ---
    st.markdown("""
----
<h2>Further Information</h2>
<ul>
  <li><a href="https://app.hedgeye.com/insights/81549-risk-report-a-quad-4-investing-playbook" target="_blank">Risk Report: A Quad 4 Investing Playbook</a></li>
  <li><a href="https://cssanalytics.wordpress.com/2025/03/20/the-growth-and-inflation-sector-timing-model/" target="_blank">The Growth and Inflation Sector Timing Model</a></li>
  <li><a href="https://simplywall.st/article/protecting-capital-with-the-all-weather-portfolio-strategy?utm_source=braze&utm_medium=email&utm_campaign=Market+Insights&utm_content=Email" target="_blank">Protecting Capital with the All-Weather Portfolio Strategy</a></li>
  <li><a href="https://www.bridgewater.com/research-and-insights/the-all-weather-story" target="_blank">The All Weather Story</a></li>
</ul>
""", unsafe_allow_html=True)

    # --- DATA PROCESSING & LIMITATIONS SECTION ---
    st.markdown("""
----
<h2>How Data is Fetched and Processed</h2>
<ul>
  <li>Firstly, all asset data is fetched from <a href='https://www.longtermtrends.net/' target='_blank'>longtermtrends.net</a> at app startup.</li>
  <li>Secondly, each dataset is resampled to business month end (BME).</li>
  <li>Thirdly, inflation data is extended: the latest official CPI data is supplemented with the <a href='https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting' target='_blank'>Cleveland Fed Nowcast</a>. The gap between the last official CPI and the nowcast is interpolated to provide a continuous series. The nowcast is an estimate and will be corrected when the next official CPI is released.</li>
  <li>Subsequently, moving averages and growth rates are calculated for both the S&P 500 and inflation data.</li>
  <li>Lastly, regimes are identified using these moving averages, and asset performance is then calculated for each regime.</li>
</ul>

<h2>Limitations</h2>
<ul>
  <li>The nowcast inflation data point is an estimation and will be corrected once official CPI numbers are released, which can impact regime identification and asset performance metrics per regime.</li>
  <li>The resampling to business month end (BME) can also impact regime identification and asset performance metrics per regime.</li>
</ul>
""", unsafe_allow_html=True)

    # --- DATA SOURCES SECTION ---
    st.markdown("""
----
<h2>Data Sources by Tab</h2>
<ul>
  <li><b>Regime Visualization Tab</b>
    <ul>
      <li>S&amp;P 500 Data: <a href="https://www.longtermtrends.net/data-sp500-since-1871/" target="_blank">longtermtrends.net/data-sp500-since-1871/</a></li>
      <li>Inflation Rate Data: <a href="https://www.longtermtrends.net/data-inflation-forecast/" target="_blank">longtermtrends.net/data-inflation-forecast/</a></li>
    </ul>
  </li>
  <li><b>Asset Classes Tab</b>
    <ul>
      <li>S&amp;P 500 Data: <a href="https://www.longtermtrends.net/data-sp500-since-1871/" target="_blank">longtermtrends.net/data-sp500-since-1871/</a></li>
      <li>Total Return Bond Index Data: <a href="https://www.longtermtrends.net/data-total-return-bond-index/" target="_blank">longtermtrends.net/data-total-return-bond-index/</a></li>
      <li>Gold Price Data: <a href="https://www.longtermtrends.net/data-gold-since-1792/" target="_blank">longtermtrends.net/data-gold-since-1792/</a></li>
    </ul>
  </li>
  <li><b>Large vs. Small Cap Tab</b>
    <ul>
      <li>MSCI USA Large Cap: <a href="https://www.longtermtrends.net/data-msci-usa-large-cap/" target="_blank">longtermtrends.net/data-msci-usa-large-cap/</a></li>
      <li>MSCI USA Mid Cap: <a href="https://www.longtermtrends.net/data-msci-usa-mid-cap/" target="_blank">longtermtrends.net/data-msci-usa-mid-cap/</a></li>
      <li>MSCI USA Small Cap: <a href="https://www.longtermtrends.net/data-msci-usa-small-cap/" target="_blank">longtermtrends.net/data-msci-usa-small-cap/</a></li>
      <li>MSCI USA Micro Cap: <a href="https://www.longtermtrends.net/data-msci-usa-micro-cap/" target="_blank">longtermtrends.net/data-msci-usa-micro-cap/</a></li>
    </ul>
  </li>
  <li><b>Cyclical vs. Defensive Tab</b>
    <ul>
      <li>MSCI USA Cyclical Stocks: <a href="https://www.longtermtrends.net/data-msci-cyclical-stocks/" target="_blank">longtermtrends.net/data-msci-cyclical-stocks/</a></li>
      <li>MSCI USA Defensive Stocks: <a href="https://www.longtermtrends.net/data-msci-defensive-stocks/" target="_blank">longtermtrends.net/data-msci-defensive-stocks/</a></li>
      <li>MSCI USA Large Cap: <a href="https://www.longtermtrends.net/data-msci-usa-large-cap/" target="_blank">longtermtrends.net/data-msci-usa-large-cap/</a></li>
      <li>MSCI USA Mid Cap: <a href="https://www.longtermtrends.net/data-msci-usa-mid-cap/" target="_blank">longtermtrends.net/data-msci-usa-mid-cap/</a></li>
      <li>MSCI USA Small Cap: <a href="https://www.longtermtrends.net/data-msci-usa-small-cap/" target="_blank">longtermtrends.net/data-msci-usa-small-cap/</a></li>
      <li>MSCI USA Micro Cap: <a href="https://www.longtermtrends.net/data-msci-usa-micro-cap/" target="_blank">longtermtrends.net/data-msci-usa-micro-cap/</a></li>
    </ul>
  </li>
  <li><b>US Sectors Tab</b>
    <ul>
      <li>US Communication Services: <a href="https://www.longtermtrends.net/data-us-communication-services/" target="_blank">longtermtrends.net/data-us-communication-services/</a></li>
      <li>US Basic Materials: <a href="https://www.longtermtrends.net/data-us-basic-materials/" target="_blank">longtermtrends.net/data-us-basic-materials/</a></li>
      <li>US Energy: <a href="https://www.longtermtrends.net/data-us-energy/" target="_blank">longtermtrends.net/data-us-energy/</a></li>
      <li>US Financial: <a href="https://www.longtermtrends.net/data-us-financial/" target="_blank">longtermtrends.net/data-us-financial/</a></li>
      <li>US Industrial: <a href="https://www.longtermtrends.net/data-us-industrial/" target="_blank">longtermtrends.net/data-us-industrial/</a></li>
      <li>US Technology: <a href="https://www.longtermtrends.net/data-us-technology/" target="_blank">longtermtrends.net/data-us-technology/</a></li>
      <li>US Consumer Staples: <a href="https://www.longtermtrends.net/data-us-consumer-staples/" target="_blank">longtermtrends.net/data-us-consumer-staples/</a></li>
      <li>US Utilities: <a href="https://www.longtermtrends.net/data-us-utiliiies/" target="_blank">longtermtrends.net/data-us-utiliiies/</a></li>
      <li>US Health Care: <a href="https://www.longtermtrends.net/data-us-thcare/" target="_blank">longtermtrends.net/data-us-thcare/</a></li>
      <li>US Consumer Discretionary: <a href="https://www.longtermtrends.net/data-us-consumer-discretionary/" target="_blank">longtermtrends.net/data-us-consumer-discretionary/</a></li>
      <li>US Real Estate: <a href="https://www.longtermtrends.net/data-us-real-estate/" target="_blank">longtermtrends.net/data-us-real-estate/</a></li>
    </ul>
  </li>
  <li><b>Factor Investing Tab</b>
    <ul>
      <li>MSCI World Momentum: <a href="https://www.longtermtrends.net/data-msci-world-momentum/" target="_blank">longtermtrends.net/data-msci-world-momentum/</a></li>
      <li>MSCI World Growth Target: <a href="https://www.longtermtrends.net/data-msci-world-growth-target/" target="_blank">longtermtrends.net/data-msci-world-growth-target/</a></li>
      <li>MSCI World Quality: <a href="https://www.longtermtrends.net/data-msci-world-quality/" target="_blank">longtermtrends.net/data-msci-world-quality/</a></li>
      <li>MSCI World Governance Quality: <a href="https://www.longtermtrends.net/data-msci-world-governance-quality/" target="_blank">longtermtrends.net/data-msci-world-governance-quality/</a></li>
      <li>MSCI World Dividend Masters: <a href="https://www.longtermtrends.net/data-msci-world-dividend-masters/" target="_blank">longtermtrends.net/data-msci-world-dividend-masters/</a></li>
      <li>MSCI World High Dividend Yield: <a href="https://www.longtermtrends.net/data-msci-world-high-dividend-yield/" target="_blank">longtermtrends.net/data-msci-world-high-dividend-yield/</a></li>
      <li>MSCI World Buyback Yield: <a href="https://www.longtermtrends.net/data-msci-world-buy-back-yield/" target="_blank">longtermtrends.net/data-msci-world-buy-back-yield/</a></li>
      <li>MSCI World Total Shareholder Yield: <a href="https://www.longtermtrends.net/data-msci-world-total-shareholder-yield/" target="_blank">longtermtrends.net/data-msci-world-total-shareholder-yield/</a></li>
      <li>MSCI World Small Cap: <a href="https://www.longtermtrends.net/data-msci-world-small-cap/" target="_blank">longtermtrends.net/data-msci-world-small-cap/</a></li>
      <li>MSCI World Equal Weighted: <a href="https://www.longtermtrends.net/data-msci-world-equal-weighted/" target="_blank">longtermtrends.net/data-msci-world-equal-weighted/</a></li>
      <li>MSCI World Enhanced Value: <a href="https://www.longtermtrends.net/data-msci-world-enhanced-value/" target="_blank">longtermtrends.net/data-msci-world-enhanced-value/</a></li>
      <li>MSCI World Prime Value: <a href="https://www.longtermtrends.net/data-msci-world-prime-value/" target="_blank">longtermtrends.net/data-msci-world-prime-value/</a></li>
      <li>MSCI World Minimum Volatility (USD): <a href="https://www.longtermtrends.net/data-msci-minimum-volatility/" target="_blank">longtermtrends.net/data-msci-minimum-volatility/</a></li>
      <li>MSCI World Risk Weighted: <a href="https://www.longtermtrends.net/data-msci-world-risk-weighted/" target="_blank">longtermtrends.net/data-msci-world-risk-weighted/</a></li>
    </ul>
  </li>
  <li><b>All-Weather Portfolio Tab</b>
    <ul>
      <li>SPDR S&amp;P 500 ETF (SPY): <a href="https://www.longtermtrends.net/data-yfin-spy/" target="_blank">longtermtrends.net/data-yfin-spy/</a></li>
      <li>Vanguard Total World Stock Index Fund ETF Shares (VT): <a href="https://www.longtermtrends.net/data-yfin-vt/" target="_blank">longtermtrends.net/data-yfin-vt/</a></li>
      <li>iShares 20+ Year Treasury Bond ETF (TLT): <a href="https://www.longtermtrends.net/data-yfin-tlt/" target="_blank">longtermtrends.net/data-yfin-tlt/</a></li>
      <li>iShares 7-10 Year Treasury Bond ETF (IEF): <a href="https://www.longtermtrends.net/data-yfin-ief/" target="_blank">longtermtrends.net/data-yfin-ief/</a></li>
      <li>iShares TIPS Bond ETF (TIP): <a href="https://www.longtermtrends.net/data-yfin-tip/" target="_blank">longtermtrends.net/data-yfin-tip/</a></li>
      <li>Invesco DB Commodity Index Tracking Fund (DBC): <a href="https://www.longtermtrends.net/data-yfin-dbc/" target="_blank">longtermtrends.net/data-yfin-dbc/</a></li>
      <li>SPDR Gold Shares (GLD): <a href="https://www.longtermtrends.net/data-yfin-gld/" target="_blank">longtermtrends.net/data-yfin-gld/</a></li>
    </ul>
  </li>
</ul>
""", unsafe_allow_html=True)