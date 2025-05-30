import hashlib
import streamlit as st
from core.constants import asset_list_tab7, asset_colors, regime_bg_colors, regime_labels_dict
from core.processing import merge_asset_with_regimes
from core.performance import generate_trade_log_df
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts
from core.asset_analysis import get_dynamic_cutoff_date_from_trade_log, render_asset_analysis_tab
from core.utils import has_gray_trades_due_to_pre_cutoff

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    min_assets_required = 7
    merged_asset_data_metrics = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    trade_log_df = generate_trade_log_df(
        merged_asset_data_metrics, sp_inflation_data, asset_list_tab7, regime_labels_dict
    )
    dynamic_cutoff_date = get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required)
    cutoff_date = dynamic_cutoff_date if dynamic_cutoff_date is not None else session_state.get('ma_start_date')
    checkbox_label = f"also include trades before {cutoff_date.strftime('%Y-%m-%d')} (when at least {min_assets_required} assets are present in a regime) in aggregations and bar charts."
    pre_cutoff_checkbox_key = f"include_pre_cutoff_trades_all_weather_{hashlib.md5(str(asset_list_tab7).encode()).hexdigest()}"

    # Show checkbox only if there are trades that would be excluded (gray) by eligibility logic
    has_excluded_trades = has_gray_trades_due_to_pre_cutoff(trade_log_df, asset_list_tab7, asset_ts_data, cutoff_date)
    if has_excluded_trades:
        include_pre_cutoff_trades = st.checkbox(
            checkbox_label,
            value=session_state.get(pre_cutoff_checkbox_key, False),
            key=pre_cutoff_checkbox_key
        )
    else:
        include_pre_cutoff_trades = False

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
