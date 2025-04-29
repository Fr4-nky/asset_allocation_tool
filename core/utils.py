import pandas as pd
import streamlit as st

def has_gray_trades_due_to_pre_cutoff(trade_log_df, asset_list, asset_ts_data, cutoff_date):
    """
    Returns True if there are trades that would be excluded (gray) due to pre-cutoff logic.
    Mirrors the trade log eligibility logic with pre_cutoff_override=False.
    """
    eligible_assets = [a for a in asset_list if a in asset_ts_data.columns]
    def is_trade_eligible_precise(row):
        asset = row.get('Asset', None)
        trade_start_date = row.get('Start Date', None)
        if isinstance(trade_start_date, str):
            try:
                trade_start_date = pd.to_datetime(trade_start_date).date()
            except Exception:
                trade_start_date = None
        if asset not in eligible_assets:
            return False
        if trade_start_date is not None and cutoff_date is not None and trade_start_date < cutoff_date and not False:
            return False
        return True
    if 'Asset' not in trade_log_df.columns or 'Start Date' not in trade_log_df.columns:
        return False
    excluded_trades = ~trade_log_df.apply(is_trade_eligible_precise, axis=1)
    return excluded_trades.any()

def has_gray_trades_due_to_late_start(trade_log_df, asset_list, asset_ts_data, cutoff_date):
    """
    Returns True if there are trades that would be excluded (gray) due to asset late start date logic.
    Mirrors the default eligible_assets logic for late assets.
    """
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list if asset in asset_ts_data.columns
    }
    eligible_assets_late = [a for a, d in asset_first_date.items() if cutoff_date is not None and d <= cutoff_date]
    if not eligible_assets_late:
        if asset_first_date:
            min_date = min(asset_first_date.values())
            eligible_assets_late = [a for a, d in asset_first_date.items() if d == min_date]
        else:
            eligible_assets_late = []
    if 'Asset' not in trade_log_df.columns:
        return False
    def is_late_asset_trade(row):
        asset = row.get('Asset', None)
        return asset not in eligible_assets_late
    gray_late_asset_trades = trade_log_df.apply(is_late_asset_trade, axis=1)
    return gray_late_asset_trades.any()
