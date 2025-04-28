import streamlit as st
from core.constants import asset_list_tab2, asset_colors, regime_bg_colors, regime_labels_dict
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.performance import generate_trade_log_df
from core.asset_analysis import render_asset_analysis_tab

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    render_asset_analysis_tab(
        tab,
        "Performance of Major Asset Classes Across Regimes",
        asset_list_tab2,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        tab_title="Asset Classes"
    )
