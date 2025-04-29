import streamlit as st
import hashlib
from core.constants import asset_list_tab3, asset_colors, regime_bg_colors, regime_labels_dict
from core.processing import merge_asset_with_regimes
from core.performance import generate_trade_log_df
from core.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.utils import has_gray_trades_due_to_pre_cutoff, has_gray_trades_due_to_late_start

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    min_assets_required = 3
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list_tab3, regime_labels_dict)
    dynamic_cutoff_date = get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required)
    cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else session_state.get('ma_start_date')
    checkbox_label = f"also include trades before {cutoff_date.strftime('%Y-%m-%d')} (when at least {min_assets_required} assets are present in a regime) in aggregations and bar charts."
    pre_cutoff_checkbox_key = f"include_pre_cutoff_trades_large_small_{hashlib.md5(str(asset_list_tab3).encode()).hexdigest()}"
    eligible_assets = [a for a in asset_list_tab3 if a in asset_ts_data.columns]
    has_excluded_trades = has_gray_trades_due_to_pre_cutoff(trade_log_df, asset_list_tab3, asset_ts_data, cutoff_date)
    if has_excluded_trades:
        include_pre_cutoff_trades = st.checkbox(
            checkbox_label,
            value=session_state.get(pre_cutoff_checkbox_key, False),
            key=pre_cutoff_checkbox_key
        )
    else:
        include_pre_cutoff_trades = False
    tab_key = f"include_late_assets_large_small_{hashlib.md5(str(asset_list_tab3).encode()).hexdigest()}"
    has_gray_late_asset_trades = has_gray_trades_due_to_late_start(trade_log_df, asset_list_tab3, asset_ts_data, cutoff_date)
    if has_gray_late_asset_trades:
        include_late_assets = st.checkbox(
            "also include assets with later start dates in aggregations and bar charts.",
            value=session_state.get(tab_key, False),
            key=tab_key
        )
    else:
        include_late_assets = False
    if not include_late_assets:
        eligible_assets = [a for a, d in {asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date() for asset in asset_list_tab3 if asset in asset_ts_data.columns}.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_list_tab3:
                min_date = min({asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date() for asset in asset_list_tab3 if asset in asset_ts_data.columns}.values())
                eligible_assets = [a for a, d in {asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date() for asset in asset_list_tab3 if asset in asset_ts_data.columns}.items() if d == min_date]
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
        include_pre_cutoff_trades=include_pre_cutoff_trades,
        include_late_assets=include_late_assets,
        cutoff_date=cutoff_date,
        eligible_assets=eligible_assets,
        tab_title="Large vs. Small Cap"
    )
