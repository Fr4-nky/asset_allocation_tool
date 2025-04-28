import streamlit as st
import hashlib
from config.constants import asset_list_tab3, asset_colors, regime_bg_colors, regime_labels_dict
from data.processing import merge_asset_with_regimes
from metrics.performance import generate_trade_log_df
from ui.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    min_assets_required = 3
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list_tab3, regime_labels_dict)
    dynamic_cutoff_date = get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required)
    cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else session_state.get('ma_start_date')
    tab_key = f"include_late_assets_large_small_{hashlib.md5(str(asset_list_tab3).encode()).hexdigest()}"
    include_late_assets = st.checkbox(
        "also include assets with later start dates in aggregations and bar charts.",
        value=session_state.get(tab_key, False),
        key=tab_key
    )
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list_tab3 if asset in asset_ts_data.columns
    }
    if not include_late_assets:
        eligible_assets = [a for a, d in asset_first_date.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_first_date:
                min_date = min(asset_first_date.values())
                eligible_assets = [a for a, d in asset_first_date.items() if d == min_date]
            else:
                eligible_assets = []
    else:
        eligible_assets = [a for a in asset_list_tab3 if a in asset_ts_data.columns]
    render_asset_analysis_tab(
        tab,
        "Performance of Large, Mid, Small, and Micro Cap Stocks Across Regimes",
        asset_list_tab3,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        include_late_assets=include_late_assets,
        eligible_assets=eligible_assets,
        tab_title="Large vs. Small Cap"
    )
