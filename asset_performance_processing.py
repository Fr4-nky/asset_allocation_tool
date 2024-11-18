# asset_performance_processing.py

import pandas as pd
import numpy as np

# Read the preprocessed asset data
asset_data = pd.read_csv('processed_data/asset_classes_preprocessed.csv', parse_dates=['DateTime'])

# Read the regimes data
regimes_data = pd.read_csv('processed_data/sp500_and_inflation_processed.csv', parse_dates=['DateTime'])

# Merge datasets based on 'DateTime', handling overlapping columns
data = pd.merge(
    asset_data,
    regimes_data,
    on='DateTime',
    how='outer',
    suffixes=('_asset', '_regime')
)

# Sort data by 'DateTime'
data.sort_values('DateTime', inplace=True)
data.reset_index(drop=True, inplace=True)

# Set 'DateTime' as index
data.set_index('DateTime', inplace=True)

# List of assets
assets = ['Gold', 'Bonds', 'S&P 500_asset']

# Compute returns for each asset
for asset in assets:
    if asset in data.columns:
        # Specify fill_method=None to prevent deprecation warning
        data[f'{asset}_Return'] = data[asset].pct_change(fill_method=None)
    else:
        print(f'Asset {asset} not found in data.')

# List of 'n' values in months
n_values = [2, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 60, 120, 180]

# Initialize list to store performance metrics
performance_metrics = []

for n in n_values:
    regime_col = f'sma_{n}_regime'
    if regime_col not in data.columns:
        print(f'Regime column {regime_col} not found in data.')
        continue  # Skip to the next n

    # Drop rows where regime_col is NaN
    data_n = data[~data[regime_col].isna()]

    for asset in assets:
        asset_return = f'{asset}_Return'
        if asset_return in data_n.columns:
            # Select data where asset_return is not NaN
            asset_data_n = data_n[[asset_return, regime_col]].dropna(subset=[asset_return])
            if not asset_data_n.empty:
                grouped = asset_data_n.groupby(regime_col)
                for regime, group in grouped:
                    avg_return = group[asset_return].mean()
                    volatility = group[asset_return].std()
                    sharpe_ratio = avg_return / volatility if volatility != 0 else np.nan
                    # Max drawdown
                    cumulative = (1 + group[asset_return]).cumprod()
                    peak = cumulative.cummax()
                    drawdown = (cumulative - peak) / peak
                    max_drawdown = drawdown.min()

                    performance_metrics.append({
                        'Asset': asset,
                        'n': n,
                        'Regime': int(regime),
                        'Average Return': avg_return,
                        'Volatility': volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Max Drawdown': max_drawdown
                    })
            else:
                print(f'No data available for {asset} to compute performance metrics for n={n}.')
        else:
            print(f'Return data for {asset} not found.')

# Convert to DataFrame
performance_df = pd.DataFrame(performance_metrics)

# Save performance_df to CSV
performance_df.to_csv('processed_data/assets_performance_by_regime.csv', index=False)
