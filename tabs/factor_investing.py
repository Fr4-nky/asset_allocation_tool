import hashlib
import streamlit as st
from core.constants import asset_list_tab6, asset_colors, regime_bg_colors, regime_labels_dict
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.processing import merge_asset_with_regimes
from core.performance import generate_trade_log_df
from core.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    min_assets_required = 5
    # Generate trade log for current MA and asset list
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list_tab6, regime_labels_dict)
    dynamic_cutoff_date = get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required)
    cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else session_state.get('ma_start_date')
    # Pre-cutoff checkbox (now appears first)
    checkbox_label = f"also include trades before {cutoff_date.strftime('%Y-%m-%d')} (when at least {min_assets_required} assets are present in a regime) in aggregations and bar charts."
    pre_cutoff_checkbox_key = f"include_pre_cutoff_trades_factor_{hashlib.md5(str(asset_list_tab6).encode()).hexdigest()}"
    include_pre_cutoff_trades = st.checkbox(
        checkbox_label,
        value=session_state.get(pre_cutoff_checkbox_key, False),
        key=pre_cutoff_checkbox_key
    )
    tab_key = f"include_late_assets_factor_{hashlib.md5(str(asset_list_tab6).encode()).hexdigest()}"
    include_late_assets = st.checkbox(
        "also include assets with later start dates in aggregations and bar charts.",
        value=session_state.get(tab_key, False),
        key=tab_key
    )
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list_tab6 if asset in asset_ts_data.columns
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
        eligible_assets = [a for a in asset_list_tab6 if a in asset_ts_data.columns]
    render_asset_analysis_tab(
        tab,
        "Performance of MSCI World Factor Strategies Across Regimes",
        asset_list_tab6,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        sp_inflation_data,
        asset_ts_data,
        include_pre_cutoff_trades=include_pre_cutoff_trades,
        include_late_assets=include_late_assets,
        cutoff_date=cutoff_date,
        eligible_assets=eligible_assets,
        tab_title="Factor Investing"
    )
