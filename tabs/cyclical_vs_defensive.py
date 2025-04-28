import streamlit as st
from config.constants import asset_list_tab4, asset_colors, regime_bg_colors, regime_labels_dict
from ui.asset_analysis import render_asset_analysis_tab

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
