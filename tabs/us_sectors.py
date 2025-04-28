import hashlib
import streamlit as st
from config.constants import asset_list_tab5, asset_colors, regime_bg_colors, regime_labels_dict
from data.processing import merge_asset_with_regimes
from ui.asset_analysis import render_asset_analysis_tab

def render(tab, asset_ts_data, sp_inflation_data, session_state):
    # Only show the second checkbox (late assets) for Sectors tab (tab 5)
    tab_key = f"include_late_assets_sectors_{hashlib.md5(str(asset_list_tab5).encode()).hexdigest()}"
    include_late_assets = st.checkbox(
        "also include assets with later start dates in aggregations and bar charts.",
        value=session_state.get(tab_key, False),
        key=tab_key
    )
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list_tab5 if asset in asset_ts_data.columns
    }
    # All 9 traces have the same start date, so this logic is simple
    cutoff_date = min(asset_first_date.values()) if asset_first_date else None
    if not include_late_assets and cutoff_date is not None:
        eligible_assets = [a for a, d in asset_first_date.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_first_date:
                min_date = min(asset_first_date.values())
                eligible_assets = [a for a, d in asset_first_date.items() if d == min_date]
            else:
                eligible_assets = []
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
