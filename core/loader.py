import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
import time

from core.fetch import fetch_and_decode

@st.cache_data
def load_data():
    # --- Define all data URLs ---
    sp500_url = "https://www.longtermtrends.net/data-sp500-since-1871/"
    inflation_url = "https://www.longtermtrends.net/data-inflation-forecast/"
    bonds_url = "https://www.longtermtrends.net/data-total-return-bond-index/"
    gold_url = "https://www.longtermtrends.net/data-gold-since-1792/"
    # MSCI USA series
    msci_large_url = "https://www.longtermtrends.net/data-msci-usa-large-cap/"
    msci_mid_url   = "https://www.longtermtrends.net/data-msci-usa-mid-cap/"
    msci_small_url = "https://www.longtermtrends.net/data-msci-usa-small-cap/"
    msci_micro_url = "https://www.longtermtrends.net/data-msci-usa-micro-cap/"
    msci_cyclical_url = "https://www.longtermtrends.net/data-msci-cyclical-stocks/"
    msci_defensive_url = "https://www.longtermtrends.net/data-msci-defensive-stocks/"
    # US Sector URLs
    comm_url = "https://www.longtermtrends.net/data-us-communication-services/"
    mat_url = "https://www.longtermtrends.net/data-us-basic-materials/"
    energy_url = "https://www.longtermtrends.net/data-us-energy/"
    financial_url = "https://www.longtermtrends.net/data-us-financial/"
    industrial_url = "https://www.longtermtrends.net/data-us-industrial/"
    technology_url = "https://www.longtermtrends.net/data-us-technology/"
    cons_stap_url = "https://www.longtermtrends.net/data-us-consumer-staples/"
    utilities_url = "https://www.longtermtrends.net/data-us-utiliiies/"
    health_url = "https://www.longtermtrends.net/data-us-thcare/"
    cons_disc_url = "https://www.longtermtrends.net/data-us-consumer-discretionary/"
    real_estate_url = "https://www.longtermtrends.net/data-us-real-estate/"
    # MSCI World Factor Strategy URLs
    world_mom_url = "https://www.longtermtrends.net/data-msci-world-momentum/"
    world_growth_url = "https://www.longtermtrends.net/data-msci-world-growth-target/"
    world_quality_url = "https://www.longtermtrends.net/data-msci-world-quality/"
    world_gov_url = "https://www.longtermtrends.net/data-msci-world-governance-quality/"
    world_div_mast_url = "https://www.longtermtrends.net/data-msci-world-dividend-masters/"
    world_high_div_url = "https://www.longtermtrends.net/data-msci-world-high-dividend-yield/"
    world_buyback_url = "https://www.longtermtrends.net/data-msci-world-buy-back-yield/"
    world_tsy_url = "https://www.longtermtrends.net/data-msci-world-total-shareholder-yield/"
    world_small_url = "https://www.longtermtrends.net/data-msci-world-small-cap/"
    world_ew_url = "https://www.longtermtrends.net/data-msci-world-equal-weighted/"
    world_enh_val_url = "https://www.longtermtrends.net/data-msci-world-enhanced-value/"
    world_prime_val_url = "https://www.longtermtrends.net/data-msci-world-prime-value/"
    world_min_vol_url = "https://www.longtermtrends.net/data-msci-minimum-volatility/"
    world_risk_url = "https://www.longtermtrends.net/data-msci-world-risk-weighted/"
    # All-Weather Portfolio URLs
    spy_url = "https://www.longtermtrends.net/data-yfin-spy/"
    vt_url = "https://www.longtermtrends.net/data-yfin-vt/"
    tlt_url = "https://www.longtermtrends.net/data-yfin-tlt/"
    ief_url = "https://www.longtermtrends.net/data-yfin-ief/"
    tip_url = "https://www.longtermtrends.net/data-yfin-tip/"
    dbc_url = "https://www.longtermtrends.net/data-yfin-dbc/"
    gld_url = "https://www.longtermtrends.net/data-yfin-gld/"

    # --- Fetch all datasets in parallel ---
    urls = {
        'S&P 500': sp500_url,
        'Inflation Rate': inflation_url,
        'Bonds': bonds_url,
        'Gold': gold_url,
        'MSCI USA Large Cap': msci_large_url,
        'MSCI USA Mid Cap': msci_mid_url,
        'MSCI USA Small Cap': msci_small_url,
        'MSCI USA Micro Cap': msci_micro_url,
        'MSCI USA Cyclical Stocks': msci_cyclical_url,
        'MSCI USA Defensive Stocks': msci_defensive_url,
        'US Communication Services': comm_url,
        'US Basic Materials': mat_url,
        'US Energy': energy_url,
        'US Financial': financial_url,
        'US Industrial': industrial_url,
        'US Technology': technology_url,
        'US Consumer Staples': cons_stap_url,
        'US Utilities': utilities_url,
        'US Health Care': health_url,
        'US Consumer Discretionary': cons_disc_url,
        'US Real Estate': real_estate_url,
        'MSCI World Momentum': world_mom_url,
        'MSCI World Growth Target': world_growth_url,
        'MSCI World Quality': world_quality_url,
        'MSCI World Governance Quality': world_gov_url,
        'MSCI World Dividend Masters': world_div_mast_url,
        'MSCI World High Dividend Yield': world_high_div_url,
        'MSCI World Buyback Yield': world_buyback_url,
        'MSCI World Total Shareholder Yield': world_tsy_url,
        'MSCI World Small Cap': world_small_url,
        'MSCI World Equal Weighted': world_ew_url,
        'MSCI World Enhanced Value': world_enh_val_url,
        'MSCI World Prime Value': world_prime_val_url,
        'MSCI World Minimum Volatility (USD)': world_min_vol_url,
        'MSCI World Risk Weighted': world_risk_url,
        'SPDR S&P 500 ETF (SPY)': spy_url,
        'Vanguard Total World Stock Index Fund ETF Shares (VT)': vt_url,
        'iShares 20+ Year Treasury Bond ETF (TLT)': tlt_url,
        'iShares 7-10 Year Treasury Bond ETF (IEF)': ief_url,
        'iShares TIPS Bond ETF (TIP)': tip_url,
        'Invesco DB Commodity Index Tracking Fund (DBC)': dbc_url,
        'SPDR Gold Shares (GLD)': gld_url
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_name = {executor.submit(fetch_and_decode, url, name): name for name, url in urls.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"ERROR fetching {name}: {e}")
                results[name] = None

    # --- Check for failed loads and warn user ---
    failed_loads = []
    for name, df in results.items():
        if df is None:
            failed_loads.append(name)
            print(f"LOADER_WARNING: Failed to load data for '{name}'.")
    if failed_loads:
        st.warning(f"Could not load data for the following assets after retries: {', '.join(failed_loads)}. They will be excluded from the analysis.")

    # Unpack results
    df_sp500 = results.get('S&P 500')
    df_inflation = results.get('Inflation Rate')
    df_bonds = results.get('Bonds')
    df_gold = results.get('Gold')
    df_msci_large = results.get('MSCI USA Large Cap')
    df_msci_mid = results.get('MSCI USA Mid Cap')
    df_msci_small = results.get('MSCI USA Small Cap')
    df_msci_micro = results.get('MSCI USA Micro Cap')
    df_msci_cyclical = results.get('MSCI USA Cyclical Stocks')
    df_msci_defensive = results.get('MSCI USA Defensive Stocks')
    df_comm = results.get('US Communication Services')
    df_mat = results.get('US Basic Materials')
    df_energy = results.get('US Energy')
    df_financial = results.get('US Financial')
    df_industrial = results.get('US Industrial')
    df_technology = results.get('US Technology')
    df_cons_stap = results.get('US Consumer Staples')
    df_utilities = results.get('US Utilities')
    df_health = results.get('US Health Care')
    df_cons_disc = results.get('US Consumer Discretionary')
    df_real_estate = results.get('US Real Estate')
    df_world_momentum = results.get('MSCI World Momentum')
    df_world_growth = results.get('MSCI World Growth Target')
    df_world_quality = results.get('MSCI World Quality')
    df_world_gov = results.get('MSCI World Governance Quality')
    df_world_div_mast = results.get('MSCI World Dividend Masters')
    df_world_high_div = results.get('MSCI World High Dividend Yield')
    df_world_buyback = results.get('MSCI World Buyback Yield')
    df_world_tsy = results.get('MSCI World Total Shareholder Yield')
    df_world_small = results.get('MSCI World Small Cap')
    df_world_ew = results.get('MSCI World Equal Weighted')
    df_world_enh_val = results.get('MSCI World Enhanced Value')
    df_world_prime_val = results.get('MSCI World Prime Value')
    df_world_min_vol = results.get('MSCI World Minimum Volatility (USD)')
    df_world_risk = results.get('MSCI World Risk Weighted')
    df_spy = results.get('SPDR S&P 500 ETF (SPY)')
    df_vt = results.get('Vanguard Total World Stock Index Fund ETF Shares (VT)')
    df_tlt = results.get('iShares 20+ Year Treasury Bond ETF (TLT)')
    df_ief = results.get('iShares 7-10 Year Treasury Bond ETF (IEF)')
    df_tip = results.get('iShares TIPS Bond ETF (TIP)')
    df_dbc = results.get('Invesco DB Commodity Index Tracking Fund (DBC)')
    df_gld = results.get('SPDR Gold Shares (GLD)')

    # Data preprocessing, resampling, merging, and filtering
    print("DEBUG: Applying resampling and merging logic...")
    today = pd.Timestamp.today().normalize()

    # 1. Resample all series to Business Month End and correct future dates
    def resample_and_correct_date(df, name):
        if df is not None:
            print(f"DEBUG: Resampling {name} (initial shape: {df.shape})")
            print(f"DEBUG: {name} last date before resample: {df.index.max()}")
            df_resampled = df.resample('BME').last()
            print(f"DEBUG: Resampled {name} shape: {df_resampled.shape}")
            print(f"DEBUG: {name} last date after resample: {df_resampled.index.max()}")
            if not df_resampled.empty and df_resampled.index.max() > today:
                print(f"WARN: {name} last date {df_resampled.index.max()} is after today {today}. Correcting...")
                original_index_name = df_resampled.index.name
                df_resampled.index = df_resampled.index.where(df_resampled.index <= today, today)
                df_resampled.index.name = original_index_name
                print(f"DEBUG: {name} last date after correction: {df_resampled.index.max()}")
            return df_resampled
        return None

    df_sp500_resampled = resample_and_correct_date(df_sp500, 'S&P 500')
    df_inflation_resampled = resample_and_correct_date(df_inflation, 'Inflation Rate')
    df_bonds_resampled = resample_and_correct_date(df_bonds, 'Bonds')
    df_gold_resampled = resample_and_correct_date(df_gold, 'Gold')

    # --- Interpolate Inflation Data to fill BME gaps ---
    df_inflation_interpolated = None
    if df_inflation_resampled is not None and not df_inflation_resampled.empty:
        print(f"DEBUG: Inflation resampled shape before interpolation: {df_inflation_resampled.shape}")
        print(f"DEBUG: Inflation resampled head before interpolation:\n{df_inflation_resampled.head()}")
        print(f"DEBUG: Inflation resampled tail before interpolation:\n{df_inflation_resampled.tail()}")
        col = df_inflation_resampled.columns[0]
        # Ensure index is datetime and sorted
        df_inflation_resampled = df_inflation_resampled.sort_index()
        # Interpolate missing values (time-based)
        df_inflation_interpolated = df_inflation_resampled.copy()
        df_inflation_interpolated[col] = df_inflation_interpolated[col].interpolate(method='time')
        print(f"DEBUG: Inflation after interpolation head:\n{df_inflation_interpolated.head()}")
        print(f"DEBUG: Inflation after interpolation tail:\n{df_inflation_interpolated.tail()}")
        # Interpolation alone is sufficient; no ffill/bfill needed
        # Confirm no NaNs remain after interpolation
        print(f"DEBUG: Inflation after interpolation (final, no ffill/bfill) head:\n{df_inflation_interpolated.head()}")
        print(f"DEBUG: Inflation after interpolation (final, no ffill/bfill) tail:\n{df_inflation_interpolated.tail()}")
        # Keep the column name as 'Inflation Rate' for merging
        df_inflation_interpolated.columns = ['Inflation Rate']
        print(f"DEBUG: Inflation interpolated columns: {df_inflation_interpolated.columns.tolist()}")


    df_msci_large_resampled = resample_and_correct_date(df_msci_large, 'MSCI USA Large Cap')
    df_msci_mid_resampled = resample_and_correct_date(df_msci_mid, 'MSCI USA Mid Cap')
    df_msci_small_resampled = resample_and_correct_date(df_msci_small, 'MSCI USA Small Cap')
    df_msci_micro_resampled = resample_and_correct_date(df_msci_micro, 'MSCI USA Micro Cap')
    df_msci_cyclical_resampled = resample_and_correct_date(df_msci_cyclical, 'MSCI USA Cyclical Stocks')
    df_msci_defensive_resampled = resample_and_correct_date(df_msci_defensive, 'MSCI USA Defensive Stocks')
    df_comm_resampled = resample_and_correct_date(df_comm, 'US Communication Services')
    df_mat_resampled = resample_and_correct_date(df_mat, 'US Basic Materials')
    df_energy_resampled = resample_and_correct_date(df_energy, 'US Energy')
    df_financial_resampled = resample_and_correct_date(df_financial, 'US Financial')
    df_industrial_resampled = resample_and_correct_date(df_industrial, 'US Industrial')
    df_technology_resampled = resample_and_correct_date(df_technology, 'US Technology')
    df_cons_stap_resampled = resample_and_correct_date(df_cons_stap, 'US Consumer Staples')
    df_utilities_resampled = resample_and_correct_date(df_utilities, 'US Utilities')
    df_health_resampled = resample_and_correct_date(df_health, 'US Health Care')
    df_cons_disc_resampled = resample_and_correct_date(df_cons_disc, 'US Consumer Discretionary')
    df_real_estate_resampled = resample_and_correct_date(df_real_estate, 'US Real Estate')
    df_world_momentum_resampled = resample_and_correct_date(df_world_momentum, 'MSCI World Momentum')
    df_world_growth_resampled = resample_and_correct_date(df_world_growth, 'MSCI World Growth Target')
    df_world_quality_resampled = resample_and_correct_date(df_world_quality, 'MSCI World Quality')
    df_world_gov_resampled = resample_and_correct_date(df_world_gov, 'MSCI World Governance Quality')
    df_world_div_mast_resampled = resample_and_correct_date(df_world_div_mast, 'MSCI World Dividend Masters')
    df_world_high_div_resampled = resample_and_correct_date(df_world_high_div, 'MSCI World High Dividend Yield')
    df_world_buyback_resampled = resample_and_correct_date(df_world_buyback, 'MSCI World Buyback Yield')
    df_world_tsy_resampled = resample_and_correct_date(df_world_tsy, 'MSCI World Total Shareholder Yield')
    df_world_small_resampled = resample_and_correct_date(df_world_small, 'MSCI World Small Cap')
    df_world_ew_resampled = resample_and_correct_date(df_world_ew, 'MSCI World Equal Weighted')
    df_world_enh_val_resampled = resample_and_correct_date(df_world_enh_val, 'MSCI World Enhanced Value')
    df_world_prime_val_resampled = resample_and_correct_date(df_world_prime_val, 'MSCI World Prime Value')
    df_world_min_vol_resampled = resample_and_correct_date(df_world_min_vol, 'MSCI World Minimum Volatility (USD)')
    df_world_risk_resampled = resample_and_correct_date(df_world_risk, 'MSCI World Risk Weighted')
    df_spy_resampled = resample_and_correct_date(df_spy, 'SPDR S&P 500 ETF (SPY)')
    df_vt_resampled = resample_and_correct_date(df_vt, 'Vanguard Total World Stock Index Fund ETF Shares (VT)')
    df_tlt_resampled = resample_and_correct_date(df_tlt, 'iShares 20+ Year Treasury Bond ETF (TLT)')
    df_ief_resampled = resample_and_correct_date(df_ief, 'iShares 7-10 Year Treasury Bond ETF (IEF)')
    df_tip_resampled = resample_and_correct_date(df_tip, 'iShares TIPS Bond ETF (TIP)')
    df_dbc_resampled = resample_and_correct_date(df_dbc, 'Invesco DB Commodity Index Tracking Fund (DBC)')
    df_gld_resampled = resample_and_correct_date(df_gld, 'SPDR Gold Shares (GLD)')

    # 2. Inner merge S&P 500 and Inflation Rate
    sp_inflation_df = pd.DataFrame()
    if df_sp500_resampled is not None and df_inflation_interpolated is not None:
        if df_sp500_resampled.index.name != df_inflation_interpolated.index.name:
            common_index_name = df_sp500_resampled.index.name or 'Date'
            df_sp500_resampled.index.name = common_index_name
            df_inflation_interpolated.index.name = common_index_name
        sp_inflation_df = pd.merge(df_sp500_resampled, df_inflation_interpolated, left_index=True, right_index=True, how='inner')
    # If only one is present, fallback as before
    elif df_sp500_resampled is not None:
        sp_inflation_df = df_sp500_resampled.copy()
        sp_inflation_df['Inflation Rate'] = np.nan
    elif df_inflation_interpolated is not None:
        sp_inflation_df = df_inflation_interpolated.copy()
        sp_inflation_df['S&P 500'] = np.nan

    elif df_sp500_resampled is not None:
        sp_inflation_df = df_sp500_resampled.copy()
        sp_inflation_df['Inflation Rate'] = np.nan
    elif df_inflation_resampled is not None:
        sp_inflation_df = df_inflation_resampled.copy()
        sp_inflation_df['S&P 500'] = np.nan

    # 3. Outer merge all assets
    asset_ts_data = pd.DataFrame()
    all_asset_dfs = [df for df in [
        df_sp500_resampled, df_gold_resampled, df_bonds_resampled,
        df_msci_large_resampled, df_msci_mid_resampled,
        df_msci_small_resampled, df_msci_micro_resampled,
        df_msci_cyclical_resampled, df_msci_defensive_resampled,
        df_comm_resampled, df_mat_resampled, df_energy_resampled,
        df_financial_resampled, df_industrial_resampled,
        df_technology_resampled, df_cons_stap_resampled,
        df_utilities_resampled, df_health_resampled,
        df_cons_disc_resampled, df_real_estate_resampled,
        df_world_momentum_resampled, df_world_growth_resampled,
        df_world_quality_resampled, df_world_gov_resampled,
        df_world_div_mast_resampled, df_world_high_div_resampled,
        df_world_buyback_resampled, df_world_tsy_resampled,
        df_world_small_resampled, df_world_ew_resampled,
        df_world_enh_val_resampled, df_world_prime_val_resampled,
        df_world_min_vol_resampled, df_world_risk_resampled,
        df_spy_resampled, df_vt_resampled, df_tlt_resampled,
        df_ief_resampled, df_tip_resampled, df_dbc_resampled,
        df_gld_resampled
    ] if df is not None]
    if len(all_asset_dfs) > 1:
        base_index_name = all_asset_dfs[0].index.name or 'Date'
        for df in all_asset_dfs:
            df.index.name = base_index_name
        asset_ts_data = all_asset_dfs[0]
        for df in all_asset_dfs[1:]:
            asset_ts_data = pd.merge(asset_ts_data, df, left_index=True, right_index=True, how='outer')
    elif len(all_asset_dfs) == 1:
        asset_ts_data = all_asset_dfs[0].copy()
    else:
        asset_ts_data = pd.DataFrame(index=pd.to_datetime([]))
        asset_ts_data.index.name = 'Date'

    # 4. Reset index to column 'DateTime'
    if not sp_inflation_df.empty:
        sp_inflation_df = sp_inflation_df.reset_index()
        date_col_sp = next((col for col in ['Date', 'index', 'DateTime'] if col in sp_inflation_df.columns), None)
        if date_col_sp and date_col_sp != 'DateTime':
            sp_inflation_df.rename(columns={date_col_sp: 'DateTime'}, inplace=True)
    if not asset_ts_data.empty:
        asset_ts_data = asset_ts_data.reset_index()
        date_col_asset = next((col for col in ['Date', 'index', 'DateTime'] if col in asset_ts_data.columns), None)
        if date_col_asset and date_col_asset != 'DateTime':
            asset_ts_data.rename(columns={date_col_asset: 'DateTime'}, inplace=True)

    # 5. Final filter (>=1972-01-01)
    filter_date = pd.Timestamp('1972-01-01')
    if 'DateTime' in sp_inflation_df.columns:
        sp_inflation_df['DateTime'] = pd.to_datetime(sp_inflation_df['DateTime'])
        sp_inflation_df = sp_inflation_df[sp_inflation_df['DateTime'] >= filter_date].copy()
    if 'DateTime' in asset_ts_data.columns:
        asset_ts_data['DateTime'] = pd.to_datetime(asset_ts_data['DateTime'])
        asset_ts_data = asset_ts_data[asset_ts_data['DateTime'] >= filter_date].copy()

    return sp_inflation_df.copy(), asset_ts_data.copy()
