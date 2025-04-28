import pandas as pd
import numpy as np

def merge_asset_with_regimes(asset_ts_df, sp_inflation_filtered):
    # Merge asset time series data with regime assignments based on DateTime.
    merged = pd.merge(asset_ts_df, sp_inflation_filtered[['DateTime', 'Regime']], on='DateTime', how='left')
    # Add Regime_Change column to mark regime transitions
    merged['Regime_Change'] = merged['Regime'].ne(merged['Regime'].shift()).cumsum()
    return merged

def compute_moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=window_size).mean()

def compute_growth(ma_data):
    return ma_data.diff()

def assign_regimes(sp_inflation_df, regime_definitions):
    sp_inflation_df['Regime'] = np.nan
    for regime in regime_definitions:
        mask = (
            (sp_inflation_df['S&P 500 MA Growth'] >= regime['S&P 500 Lower']) &
            (sp_inflation_df['S&P 500 MA Growth'] < regime['S&P 500 Upper']) &
            (sp_inflation_df['Inflation Rate MA Growth'] >= regime['Inflation Lower']) &
            (sp_inflation_df['Inflation Rate MA Growth'] < regime['Inflation Upper'])
        )
        sp_inflation_df.loc[mask, 'Regime'] = regime['Regime']
    return sp_inflation_df
