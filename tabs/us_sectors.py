import hashlib
import streamlit as st
import pandas as pd
from core.constants import asset_list_tab5, asset_colors, regime_bg_colors, regime_labels_dict
from core.processing import merge_asset_with_regimes
from core.asset_analysis import render_asset_analysis_tab
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.performance import generate_trade_log_df

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    # Only show the second checkbox (late assets) for Sectors tab (tab 5)
    tab_key = f"include_late_assets_sectors_{hashlib.md5(str(asset_list_tab5).encode()).hexdigest()}"
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list_tab5 if asset in asset_ts_data.columns
    }
    # All 9 traces have the same start date, so this logic is simple
    cutoff_date = min(asset_first_date.values()) if asset_first_date else None
    # Determine eligible assets for default (not including late assets)
    eligible_assets_late = [a for a, d in asset_first_date.items() if d <= cutoff_date] if cutoff_date is not None else []
    if not eligible_assets_late:
        if asset_first_date:
            min_date = min(asset_first_date.values())
            eligible_assets_late = [a for a, d in asset_first_date.items() if d == min_date]
        else:
            eligible_assets_late = []
    # Find gray trades due to late start dates (asset not in eligible_assets_late)
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list_tab5, regime_labels_dict)
    def is_late_asset_trade(row):
        asset = row.get('Asset', None)
        return asset not in eligible_assets_late
    gray_late_asset_trades = trade_log_df.apply(is_late_asset_trade, axis=1)
    has_gray_late_asset_trades = gray_late_asset_trades.any()
    if has_gray_late_asset_trades:
        include_late_assets = st.checkbox(
            "also include assets with later start dates in aggregations and bar charts.",
            value=session_state.get(tab_key, False),
            key=tab_key
        )
    else:
        include_late_assets = False
    if not include_late_assets:
        eligible_assets = eligible_assets_late
    else:
        eligible_assets = [a for a in asset_list_tab5 if a in asset_ts_data.columns]
    render_asset_analysis_tab(
        tab,
        "Performance of US Sector ETFs Across Regimes",
        asset_list_tab5,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        include_late_assets=include_late_assets,
        eligible_assets=eligible_assets,
        tab_title="US Sectors"
    )
