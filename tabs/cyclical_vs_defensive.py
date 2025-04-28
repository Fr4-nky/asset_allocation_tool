import streamlit as st
from core.constants import asset_list_tab4, asset_colors, regime_bg_colors, regime_labels_dict
from core.asset_analysis import render_asset_analysis_tab
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    render_asset_analysis_tab(
        tab,
        "Cyclical vs. Defensive Stocks Across Regimes",
        asset_list_tab4,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        tab_title="Cyclical vs. Defensive"
    )
