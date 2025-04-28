import hashlib
import streamlit as st
from config.constants import asset_list_tab7, asset_colors, regime_bg_colors, regime_labels_dict
from data.processing import merge_asset_with_regimes
from metrics.performance import generate_trade_log_df
from ui.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    min_assets_required = 7
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(
        merged_asset_data_metrics, sp_inflation_data, asset_list_tab7, regime_labels_dict
    )
    dynamic_cutoff_date = get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required)
    cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else session_state.get('ma_start_date')
    pre_cutoff_checkbox_key = f"include_pre_cutoff_trades_all_weather_{hashlib.md5(str(asset_list_tab7).encode()).hexdigest()}"
    checkbox_label = "include trades before cutoff date (for assets with later start dates) in aggregations and bar charts."
    include_pre_cutoff_trades = st.checkbox(
        checkbox_label,
        value=session_state.get(pre_cutoff_checkbox_key, False),
        key=pre_cutoff_checkbox_key
    )
    render_asset_analysis_tab(
        tab,
        "Performance of All-Weather Portfolio Across Regimes",
        asset_list_tab7,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        include_pre_cutoff_trades=include_pre_cutoff_trades,
        include_late_assets=True,
        cutoff_date=cutoff_date,
        tab_title="All-Weather Portfolio"
    )
