import pandas as pd
import numpy as np

def generate_trade_log(merged_asset_data_metrics, asset_list, regime_labels_dict=None):
    trade_log_results = []
    regime_changes = merged_asset_data_metrics['Regime'].ne(merged_asset_data_metrics['Regime'].shift()).cumsum()
    for asset in asset_list:
        if asset not in merged_asset_data_metrics.columns:
            continue
        for (regime, segment), group in merged_asset_data_metrics[['DateTime', asset, 'Regime']].dropna().groupby(['Regime', regime_changes]):
            if pd.isna(regime) or len(group) < 1:
                continue
            start_date = group['DateTime'].iloc[0]
            end_date = group['DateTime'].iloc[-1]
            price_start = group[asset].iloc[0]
            price_end = group[asset].iloc[-1]
            period_return = (price_end - price_start) / price_start if price_start != 0 else float('nan')
            if len(group) >= 2:
                returns = group[asset].pct_change().dropna()
                volatility = returns.std() * np.sqrt(12) if not returns.empty else np.nan
                cumulative = (1 + returns).cumprod()
                cumulative_max = cumulative.cummax()
                drawdown = cumulative / cumulative_max - 1
                max_drawdown = drawdown.min() if not drawdown.empty else np.nan
                sharpe_ratio = period_return / volatility if volatility != 0 and not np.isnan(volatility) and not np.isnan(period_return) else np.nan
            else:
                volatility = np.nan
                sharpe_ratio = np.nan
                max_drawdown = np.nan
            trade_log_results.append({
                'Asset': asset,
                'Regime': regime_labels_dict.get(regime, str(regime)) if regime_labels_dict else str(regime),
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Start Price': price_start,
                'End Price': price_end,
                'Period Return': period_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown
            })
    df = pd.DataFrame(trade_log_results)
    if not df.empty:
        df['Regime'] = df['Regime'].astype(str)
    return df

def generate_aggregated_metrics(trade_log_df, merged_asset_data_metrics, asset_list, regime_labels_dict):
    required_cols = {'Period Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'}
    avg_metrics = []
    if not required_cols.issubset(set(trade_log_df.columns)):
        return pd.DataFrame()
    for (regime, asset), group in trade_log_df.groupby(['Regime', 'Asset'], observed=False):
        monthly_returns = []
        for idx, row in group.iterrows():
            start_date = pd.to_datetime(row['Start Date'])
            end_date = pd.to_datetime(row['End Date'])
            asset_prices = merged_asset_data_metrics[['DateTime', asset]].dropna()
            asset_prices = asset_prices[(asset_prices['DateTime'] >= start_date) & (asset_prices['DateTime'] <= end_date)]
            if not asset_prices.empty:
                returns = asset_prices[asset].pct_change().dropna()
                monthly_returns.append(returns)
        if monthly_returns:
            all_returns = pd.concat(monthly_returns)
            mean_monthly = all_returns.mean()
            std_monthly = all_returns.std()
            ann_return = mean_monthly * 12
            ann_vol = std_monthly * (12 ** 0.5)
            sharpe = ann_return / ann_vol if ann_vol > 0 else float('nan')
        else:
            ann_return = float('nan')
            ann_vol = float('nan')
            sharpe = float('nan')
        avg_drawdown = group['Max Drawdown'].mean()
        avg_metrics.append({
            'Regime': str(regime),
            'Asset': asset,
            'Annualized Return (Aggregated)': ann_return,
            'Annualized Volatility (Aggregated)': ann_vol,
            'Sharpe Ratio (Aggregated)': sharpe,
            'Average Max Drawdown (Period Avg)': avg_drawdown,
        })
    avg_metrics_table = pd.DataFrame(avg_metrics)
    if not avg_metrics_table.empty:
        avg_metrics_table['Regime'] = avg_metrics_table['Regime'].map(lambda r: regime_labels_dict.get(r, str(r)))
        regime_order = list(regime_labels_dict.values())
        avg_metrics_table['Regime'] = pd.Categorical(avg_metrics_table['Regime'], categories=regime_order, ordered=True)
        avg_metrics_table.sort_values(['Regime', 'Asset'], inplace=True)
    return avg_metrics_table
