import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pandas as pd  # for regime shading logic
import numpy as np

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
    # Reset zoom on each render
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    st.plotly_chart(fig, use_container_width=False, key=f"norm_chart_{chart_title}_{id(fig)}")

def plot_metrics_bar_charts(avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict, tab_title=""):
    metrics_to_display = ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Sharpe Ratio (Aggregated)', 'Average Max Drawdown (Period Avg)']
    st.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Bar Charts of Performance Metrics</h2>
    """, unsafe_allow_html=True)

    # Ensure 'Regime' is categorical and ordered if needed
    if not pd.api.types.is_categorical_dtype(avg_metrics_table['Regime']):
        avg_metrics_table['Regime'] = avg_metrics_table['Regime'].astype('category')
    # Explicitly get all categories AFTER ensuring it's a category type
    all_regimes = avg_metrics_table['Regime'].cat.categories

    for metric in metrics_to_display:
        fig3 = go.Figure()

        # --- Add Regime Background Shading FIRST ---
        # Create a mapping from regime category index to regime name/label
        regime_idx_to_name = {i: name for i, name in enumerate(all_regimes)}
        for i, regime_name in enumerate(all_regimes):
            # Find the original numeric key for this regime label
            regime_numeric_key = next((k for k, v in regime_labels_dict.items() if v == regime_name), None)
            if regime_numeric_key is not None:
                color = regime_bg_colors.get(regime_numeric_key, 'rgba(200,200,200,0.10)')
            else:
                color = 'rgba(200,200,200,0.10)' # Fallback color
            fig3.add_vrect(
                x0=i - 0.5,
                x1=i + 0.5,
                fillcolor=color,
                opacity=1.0,
                layer="below",
                line_width=0
            )

        # --- Pivot Data for Consistent Grouping ---
        try:
            pivot_table = avg_metrics_table.pivot_table(
                index='Regime',
                columns='Asset',
                values=metric,
                observed=True # Address FutureWarning and use future default
            )
            # Ensure all regimes are present in the index, even if they have no data for this metric/asset combo
            pivot_table = pivot_table.reindex(all_regimes, fill_value=np.nan)
        except Exception as e:
            st.error(f"Error pivoting data for metric '{metric}': {e}")

            continue # Skip this metric if pivoting fails

        # --- Add Bar Traces based on Pivoted Data ---
        for asset_name in pivot_table.columns: # Iterate through assets (columns of pivot table)
            fig3.add_trace(go.Bar(
                x=pivot_table.index, # Use the common regime index for x-axis
                y=pivot_table[asset_name], # Use the metric values for this asset
                name=asset_name,
                marker_color=asset_colors.get(asset_name, 'gray')
            ))

        fig3.update_layout(
            barmode='group',
            xaxis_title='',
            yaxis_title=metric,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(all_regimes))), # Use indices 0, 1, 2...
                ticktext=list(all_regimes) # Display the regime names
            ),
            title={
                'text': metric + ' by Asset and Regime',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            },
            width=800,
            height=500
        )
        # Reset zoom on bar charts
        fig3.update_xaxes(autorange=True)
        fig3.update_yaxes(autorange=True)
        if metric in ['Annualized Return (Aggregated)', 'Annualized Volatility (Aggregated)', 'Average Max Drawdown (Period Avg)']:
            fig3.update_yaxes(tickformat=".0%")

        # Use unique key including metric AND tab_title to avoid conflicts across tabs
        safe_tab_title = tab_title.replace(' ', '_').replace('/', '_') # Make tab title safe for key
        st.plotly_chart(fig3, use_container_width=False, key=f"bar_chart_{safe_tab_title}_{metric.replace(' ', '_')}_{id(fig3)}")
