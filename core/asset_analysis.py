import pandas as pd
import datetime
import re
import streamlit as st
from core.processing import merge_asset_with_regimes
from core.performance import generate_trade_log_df, generate_aggregated_metrics
from core.charts import plot_asset_performance_over_time, plot_metrics_bar_charts

# --- Helper: Compute dynamic cutoff date based on trade log (regime start with N assets) ---
def get_dynamic_cutoff_date_from_trade_log(trade_log_df, min_assets_required):
    # For each date (not regime), count number of unique assets present in the trade log
    trade_log_df = trade_log_df.sort_values('Start Date')
    asset_counts = trade_log_df.groupby('Start Date')['Asset'].nunique()
    eligible_dates = asset_counts[asset_counts >= min_assets_required]
    if not eligible_dates.empty:
        return pd.to_datetime(eligible_dates.index[0]).date()
    else:
        return None

# --- Main Tab Rendering Logic ---
def render_asset_analysis_tab(tab, title, asset_list, asset_colors, regime_bg_colors, regime_labels_dict, sp_inflation_data, asset_ts_data, include_pre_cutoff_trades=False, include_late_assets=False, cutoff_date=None, eligible_assets=None, tab_title=""):
    tab.markdown(f"""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>{title}</h2>
    """, unsafe_allow_html=True)
    # Use the passed include_late_assets argument in filtering logic
    with st.spinner('Merging asset data with regimes...'):
        merged_asset_data = merge_asset_with_regimes(asset_ts_data, sp_inflation_data)
    # Compute regime_periods for consistent chart shading and trade log
    df_periods = sp_inflation_data.dropna(subset=['Regime'])
    df_periods = df_periods[df_periods['Regime'] != 'Unknown']
    change_mask2 = df_periods['Regime'].ne(df_periods['Regime'].shift())
    df_start2 = df_periods.loc[change_mask2, ['DateTime', 'Regime']].copy()
    # Format start dates
    df_start2['Start'] = df_start2['DateTime']
    df_start2['End'] = df_start2['Start'].shift(-1)
    df_start2.at[df_start2.index[-1], 'End'] = df_periods['DateTime'].iloc[-1]
    df_start2['Regime'] = df_start2['Regime'].map(regime_labels_dict)
    regime_periods = df_start2[['Regime', 'Start', 'End']].to_dict(orient='records')
    # Set xaxis range for normalized asset charts if tab-specific cutoff is used
    xaxis_range = None
    from core.constants import asset_list_tab3, asset_list_tab6
    if asset_list == asset_list_tab3:
        xaxis_range = ["1994-06-30", None]
    elif asset_list == asset_list_tab6:
        xaxis_range = ["1977-03-31", None]
    plot_asset_performance_over_time(
        merged_asset_data,
        asset_list,
        asset_colors,
        regime_bg_colors,
        regime_labels_dict,
        title + ' (Normalized to 100 at First Available Date)',
        regime_periods=regime_periods,
        xaxis_range=xaxis_range
    )
    tab.markdown("""
    <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Trade Log</h2>
    """, unsafe_allow_html=True)
    merged_asset_data_metrics = merged_asset_data.copy()
    trade_log_df = generate_trade_log_df(merged_asset_data_metrics, sp_inflation_data, asset_list, regime_labels_dict)
    # --- Asset filtering for aggregated metrics/bar charts ---
    # Compute first available date for each asset
    asset_first_date = {
        asset: asset_ts_data.loc[asset_ts_data[asset].notna(), 'DateTime'].min().date()
        for asset in asset_list if asset in asset_ts_data.columns
    }
    print(f"[DEBUG] asset_first_date for tab '{title}':")
    for asset, date in asset_first_date.items():
        print(f"    {asset}: {date}")
    # Use the tab-specific include_late_assets value
    passed_cutoff_date = cutoff_date # Rename argument to avoid confusion
    from core.constants import asset_list_tab3 # Remove asset_list_tab6 import here

    if passed_cutoff_date is not None:
        cutoff_date = passed_cutoff_date # Use the date calculated externally and passed in
        print(f"[DEBUG] Using passed cutoff_date for tab '{title}': {cutoff_date}")
    elif asset_list == asset_list_tab3:
        cutoff_date = datetime.date(1994, 6, 30) # Hardcoded for Tab 3 (Large vs Small)
        print(f"[DEBUG] Using hardcoded cutoff_date for tab '{title}': {cutoff_date}")
    # REMOVED: elif asset_list == asset_list_tab6: condition
    else:
        cutoff_date = st.session_state.get('ma_start_date') # Fallback to MA start date
        print(f"[DEBUG] Using fallback cutoff_date (ma_start_date) for tab '{title}': {cutoff_date}")

    print(f"[DEBUG] Final cutoff_date being used for filtering in tab '{title}': {cutoff_date}")
    print(f"[DEBUG] include_late_assets for tab '{title}': {include_late_assets}")

    if not include_late_assets and cutoff_date is not None:
        eligible_assets = [a for a, d in asset_first_date.items() if d <= cutoff_date]
        if not eligible_assets:
            if asset_first_date:
                min_date = min(asset_first_date.values())
                eligible_assets = [a for a, d in asset_first_date.items() if d == min_date]
            else:
                eligible_assets = []
    else:
        eligible_assets = [a for a in asset_list if a in asset_ts_data.columns]
    print(f"[DEBUG] eligible_assets for tab '{title}': {eligible_assets}")

    # --- Central eligibility function for trade inclusion ---
    def is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override):
        asset = row.get('Asset', None)
        trade_start_date = row.get('Start Date', None)
        # Try to parse trade_start_date as date if it's a string
        if isinstance(trade_start_date, str):
            try:
                trade_start_date = pd.to_datetime(trade_start_date).date()
            except Exception:
                trade_start_date = None
        if asset not in eligible_assets:
            return False
        if trade_start_date is not None and trade_start_date < cutoff_date and not pre_cutoff_override:
            return False
        return True

    # Determine pre_cutoff_override for this tab
    pre_cutoff_override = include_pre_cutoff_trades

    # --- AGGREGATION: filter trades for metrics/bar charts based on eligibility function ---
    if 'Asset' in trade_log_df:
        trade_log_df = trade_log_df[trade_log_df['Asset'].isin(asset_list)].copy()
    else:
        trade_log_df = trade_log_df.copy()
    eligible_mask = trade_log_df.apply(lambda row: is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override), axis=1)
    filtered_trade_log_df = trade_log_df[eligible_mask].copy()

    # --- Trade log highlighting uses the same function ---
    def highlight_trade(row):
        if not is_trade_eligible(row, eligible_assets, cutoff_date, pre_cutoff_override):
            return ['background-color: #e0e0e0'] * len(row)
        # Otherwise, use regime background color
        regime_label = row.get('Regime', None)
        regime_num = None
        for k, v in regime_labels_dict.items():
            if v == regime_label:
                regime_num = k
                break
        css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
        if css_rgba.startswith('rgba'):
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
            if match:
                r,g,b,_ = match.groups()
                from core.constants import REGIME_BG_ALPHA
                color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
            else:
                color = f"rgba(200,200,200,{REGIME_BG_ALPHA})" # Fallback
        else:
            color = '#eeeeee'
        return [f'background-color: {color}'] * len(row)

    tab.dataframe(
        trade_log_df.style
            .format({
                'Start Price': '{:.2f}',
                'End Price': '{:.2f}',
                'Period Return': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}'
            })
            .apply(highlight_trade, axis=1),
        use_container_width=True
    )
    # --- FOOTNOTES for Trade Log ---
    tab.markdown(r"""
    **Column Definitions:**
    - **Period Return**: (End Price - Start Price) / Start Price
    - **Volatility**: Standard deviation of monthly returns within the period, annualized (multiplied by $\sqrt{12}$)
    - **Sharpe Ratio**: Annualized mean monthly return divided by annualized volatility, assuming risk-free rate = 0
    - **Max Drawdown**: Maximum observed loss from a peak to a trough during the period, based on monthly closing prices (as a percentage of the peak)
    
    *Volatility and Sharpe ratio cannot be calculated for 1-month periods.*
    """, unsafe_allow_html=True)

    # Keep conditional footnote from upstream
    from core.constants import asset_list_tab3, asset_list_tab5, asset_list_tab6
    if asset_list in [asset_list_tab3, asset_list_tab5, asset_list_tab6]:
        tab.markdown(
            '*If the background color is gray, the trade is not included in the aggregations and the bar charts.*',
            unsafe_allow_html=True
        )

    # --- AGGREGATED METRICS TABLE --- (Keep title, use upstream logic for avg_metrics_table)

    show_aggregated_metrics = st.session_state.is_premium_user
    print(f"[DEBUG] show_aggregated_metrics for tab '{title}': {show_aggregated_metrics}")
    tab.markdown("""
        <h2 style='text-align:left; font-size:2.0rem; font-weight:600;'>Aggregated Performance Metrics</h2>
        """, unsafe_allow_html=True)
    if show_aggregated_metrics:


        avg_metrics_table = generate_aggregated_metrics(filtered_trade_log_df, merged_asset_data_metrics, eligible_assets, regime_labels_dict)
        avg_metrics_table = avg_metrics_table[avg_metrics_table['Regime'] != 'Unknown']
        avg_metrics_table = avg_metrics_table[avg_metrics_table['Asset'].isin(eligible_assets)]
        avg_metrics_table = avg_metrics_table.reset_index(drop=True)
        regime_order = [regime_labels_dict[k] for k in [2,1,4,3]]
        asset_order = eligible_assets  # Use eligible assets for ordering

        avg_metrics_table['Regime'] = pd.Categorical(avg_metrics_table['Regime'], categories=regime_order, ordered=True)
        avg_metrics_table['Asset'] = pd.Categorical(avg_metrics_table['Asset'], categories=asset_order, ordered=True)
        avg_metrics_table = avg_metrics_table.sort_values(['Regime','Asset']).reset_index(drop=True)

        def highlight_regime_avg(row):
            regime_label = row['Regime']
            regime_num = None
            for k, v in regime_labels_dict.items():
                if v == regime_label:
                    regime_num = k
                    break
            css_rgba = regime_bg_colors.get(regime_num, '#eeeeee')
            if css_rgba.startswith('rgba'):
                match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', css_rgba)
                if match:
                    r, g, b, _ = match.groups()
                    from core.constants import REGIME_BG_ALPHA
                    color = f"rgba({r},{g},{b},{REGIME_BG_ALPHA})"
                else:
                    color = f"rgba(200,200,200,{REGIME_BG_ALPHA})"  # Fallback
            else:
                color = '#eeeeee'
            return [f'background-color: {color}'] * len(row)

        tab.dataframe(
            avg_metrics_table.style
                .format({
                    'Annualized Return (Aggregated)': '{:.2%}',
                    'Annualized Volatility (Aggregated)': '{:.2%}',
                    'Sharpe Ratio (Aggregated)': '{:.2f}',
                    'Average Max Drawdown (Period Avg)': '{:.2%}'
                })
                .apply(highlight_regime_avg, axis=1),
            use_container_width=True
        )

        tab.markdown("""
        **Aggregation & Calculation Notes:**
        - **Annualized Return (Aggregated):** Average of monthly returns for each regime-asset group, annualized by multiplying by 12.
        - **Annualized Volatility (Aggregated):** Standard deviation of those monthly returns, annualized by multiplying by √12.
        - **Sharpe Ratio (Aggregated):** Aggregated annual return divided by aggregated annual volatility (0% risk-free rate).
        - **Average Max Drawdown:** Mean of the maximum drawdowns observed in each period for each regime-asset group.
        - **Missing Data Handling:** Excludes any missing (NaN) values from all calculations.
        """, unsafe_allow_html=True)

        plot_metrics_bar_charts(avg_metrics_table, asset_colors, regime_bg_colors, regime_labels_dict, tab_title)
    else:
        from config import API_BASE_URL
        # Create proper redirection to the community page
        tab.markdown(f"""
        <div style="padding: 2rem; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #F6F6F6; border-radius: 0.5rem; height: 24rem; margin: 1rem 0;">
            <div style="margin-bottom: 1rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
            </div>
            <div style="font-weight: 300; text-align: center; margin-bottom: 1.5rem;">
               Unlock aggregated performance metrics!
            </div>
            <a href="{API_BASE_URL}/community/" target="_top"
               style="background-color: #2563eb; color: white; padding: 0.75rem 2.5rem; border-radius: 0.75rem; font-weight: 300; text-decoration: none; display: inline-block; cursor: pointer;">
                Become a Member
            </a>
        </div>
        """, unsafe_allow_html=True)
