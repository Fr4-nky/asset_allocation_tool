import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pandas as pd  # for regime shading logic

def plot_asset_performance_over_time(merged_asset_data, asset_list, asset_colors, regime_bg_colors, regime_labels_dict, chart_title, regime_periods=None, xaxis_range=None):
    fig = go.Figure()
    # Sort data
    merged_asset_data = merged_asset_data.sort_values("DateTime")
    # Add regime background shading, using provided periods or computed segments
    shapes = []
    if regime_periods is not None:
        # regime_periods: list of dicts with keys 'Regime', 'Start', 'End'
        for seg in regime_periods:
            regime = seg['Regime']
            # Accept both string and numeric regime keys
            regime_key = next((k for k, v in regime_labels_dict.items() if v == regime or k == regime), regime)
            if pd.isna(regime_key) or regime_key == 'Unknown':
                continue
            color = regime_bg_colors.get(regime_key, 'rgba(200,200,200,0.10)')
            # Convert pandas Timestamp to python datetime or string for plotly
            x0 = seg['Start']
            x1 = seg['End']
            if hasattr(x0, 'to_pydatetime'):
                x0 = x0.to_pydatetime()
            if hasattr(x1, 'to_pydatetime'):
                x1 = x1.to_pydatetime()
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1, fillcolor=color, opacity=1.0, layer="below", line_width=0
            ))
    else:
        # fallback to grouping logic
        regime_changes = merged_asset_data['Regime'].ne(merged_asset_data['Regime'].shift()).cumsum()
        for (regime, segment), group in merged_asset_data.groupby(['Regime', regime_changes]):
            if pd.isna(regime) or regime == 'Unknown':
                continue
            color = regime_bg_colors.get(regime, 'rgba(200,200,200,0.10)')
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=group['DateTime'].iloc[0], x1=group['DateTime'].iloc[-1],
                y0=0, y1=1, fillcolor=color, opacity=1.0, layer="below", line_width=0
            ))
    fig.update_layout(shapes=shapes)
    # Log regimes for debugging
    print("DEBUG: plot_asset_performance_over_time regimes:", merged_asset_data['Regime'].value_counts().to_dict())
    # Plot normalized asset lines
    for asset in asset_list:
        if asset not in merged_asset_data.columns:
            continue
        asset_series = merged_asset_data[['DateTime', asset]].dropna()
        if asset_series.empty:
            continue
        norm = asset_series[asset] / asset_series[asset].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=asset_series['DateTime'],
            y=norm,
            mode='lines',
            name=asset,
            line=dict(color=asset_colors.get(asset, 'gray'), width=2)
        ))
    fig.update_layout(
        title={
            'text': chart_title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Date', range=xaxis_range),
        yaxis=dict(title='Normalized Price', type='linear'),
        hovermode='x unified',
        width=1200,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black")
    )
    st.plotly_chart(fig, use_container_width=False, key=f"norm_chart_{chart_title}_{id(fig)}")

def plot_metrics_bar_charts(avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict, asset_list=None):
    metrics_to_display = ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Sharpe Ratio (Aggregated)', 'Average Max Drawdown (Period Avg)']
    st.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Bar Charts of Performance Metrics</h2>
    """, unsafe_allow_html=True)
    # Checkbox is now only shown once per tab, in the main app file
    for metric in metrics_to_display:
        fig3 = go.Figure()
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
        if metric in ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Average Max Drawdown (Period Avg)']:
            fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=False, key=f"bar_chart_{metric}_{id(fig3)}")
