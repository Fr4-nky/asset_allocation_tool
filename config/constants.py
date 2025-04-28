# Color maps for assets and regimes
asset_colors = {
    'S&P 500': 'blue',
    'Gold': 'gold',
    'Bonds': 'black',
    'Inflation Rate': 'red',
    'S&P 500 MA': 'blue',
    'Inflation Rate MA': 'red',
    'MSCI USA Large Cap': 'purple',
    'MSCI USA Mid Cap': 'orange',
    'MSCI USA Small Cap': 'lime',
    'MSCI USA Micro Cap': 'cyan',
    'MSCI USA Cyclical Stocks': 'magenta',
    'MSCI USA Defensive Stocks': 'teal',
    # US Sector ETFs
    'US Communication Services': '#2E8B57',
    'US Basic Materials': '#8B0000',
    'US Energy': '#FF8C00',
    'US Financial': '#FFD700',
    'US Industrial': '#1E90FF',
    'US Technology': '#6A5ACD',
    'US Consumer Discretionary': '#FF69B4',
    'US Consumer Staples': '#D2691E',
    'US Utilities': '#00CED1',
    'US Health Care': '#DC143C',
    'US Real Estate': '#8B4513',
    # MSCI World Factor Strategies
    'MSCI World Momentum': '#1b9e77',
    'MSCI World Growth Target': '#d95f02',
    'MSCI World Quality': '#7570b3',
    'MSCI World Governance Quality': '#e7298a',
    'MSCI World Dividend Masters': '#66a61e',
    'MSCI World High Dividend Yield': '#e6ab02',
    'MSCI World Buyback Yield': '#a6761d',
    'MSCI World Total Shareholder Yield': '#666666',
    'MSCI World Small Cap': '#1f78b4',
    'MSCI World Equal Weighted': '#b2df8a',
    'MSCI World Enhanced Value': '#fb9a99',
    'MSCI World Prime Value': '#fdbf6f',
    'MSCI World Minimum Volatility (USD)': '#cab2d6',
    'MSCI World Risk Weighted': '#6a3d9a',
    # Add All-Weather Portfolio asset colors
    'SPDR S&P 500 ETF (SPY)': 'blue',
    'Vanguard Total World Stock Index Fund ETF Shares (VT)': 'green',
    'iShares 20+ Year Treasury Bond ETF (TLT)': 'black',
    'iShares 7-10 Year Treasury Bond ETF (IEF)': 'gray',
    'iShares TIPS Bond ETF (TIP)': 'red',
    'Invesco DB Commodity Index Tracking Fund (DBC)': 'orange',
    'SPDR Gold Shares (GLD)': 'gold'
}

REGIME_BG_ALPHA = 0.13  # Use the same alpha for all charts and tables
regime_bg_colors = {
    1: f'rgba(255, 0, 0, {REGIME_BG_ALPHA})',     # Reflation - Red
    2: f'rgba(0, 255, 0, {REGIME_BG_ALPHA})',     # Goldilocks - Green
    3: f'rgba(255, 255, 0, {REGIME_BG_ALPHA})',   # Stagflation - Yellow
    4: f'rgba(0, 128, 255, {REGIME_BG_ALPHA})',   # Deflation - Blue
    'Unknown': 'rgba(220, 220, 220, 0.5)'
}

regime_legend_colors = {
    1: 'rgba(255, 0, 0, 0.7)',    # Reflation - Red
    2: 'rgba(0, 255, 0, 0.7)',   # Goldilocks - Green
    3: 'rgba(255, 255, 0, 0.7)', # Stagflation - Yellow
    4: 'rgba(0, 128, 255, 0.7)', # Deflation - Blue
    'Unknown': 'rgba(220, 220, 220, 0.5)'
}

regime_labels_dict = {
    1: '🎈 Reflation: Rising growth, rising inflation',
    2: '👧🏼 Goldilocks: Rising growth, falling inflation',
    3: '✋ Stagflation: Falling growth, rising inflation',
    4: '💨 Deflation: Falling growth, falling inflation',
    'Unknown': 'Unknown'
}

asset_list_tab2 = ['S&P 500', 'Bonds', 'Gold']
asset_list_tab3 = [
    'MSCI USA Large Cap',
    'MSCI USA Mid Cap',
    'MSCI USA Small Cap',
    'MSCI USA Micro Cap'
]

asset_list_tab4 = [
    'MSCI USA Cyclical Stocks',
    'MSCI USA Defensive Stocks'
]

asset_list_tab5 = [
    'US Communication Services',
    'US Basic Materials',
    'US Energy',
    'US Financial',
    'US Industrial',
    'US Technology',
    'US Consumer Discretionary',
    'US Consumer Staples',
    'US Utilities',
    'US Health Care',
    'US Real Estate'
]

asset_list_tab6 = [
    'MSCI World Momentum',
    'MSCI World Growth Target',
    'MSCI World Quality',
    'MSCI World Governance Quality',
    'MSCI World Dividend Masters',
    'MSCI World High Dividend Yield',
    'MSCI World Buyback Yield',
    'MSCI World Total Shareholder Yield',
    'MSCI World Small Cap',
    'MSCI World Equal Weighted',
    'MSCI World Enhanced Value',
    'MSCI World Prime Value',
    'MSCI World Minimum Volatility (USD)',
    'MSCI World Risk Weighted'
]

asset_list_tab7 = [
    'SPDR S&P 500 ETF (SPY)',
    'Vanguard Total World Stock Index Fund ETF Shares (VT)',
    'iShares 20+ Year Treasury Bond ETF (TLT)',
    'iShares 7-10 Year Treasury Bond ETF (IEF)',
    'iShares TIPS Bond ETF (TIP)',
    'Invesco DB Commodity Index Tracking Fund (DBC)',
    'SPDR Gold Shares (GLD)'
]

regime_definitions = [
    {
        'Regime': 1,
        'S&P 500 Lower': 0,
        'S&P 500 Upper': float('inf'),
        'Inflation Lower': 0,
        'Inflation Upper': float('inf'),
        'Label': 'Rising Growth & Rising Inflation'
    },
    {
        'Regime': 2,
        'S&P 500 Lower': 0,
        'S&P 500 Upper': float('inf'),
        'Inflation Lower': float('-inf'),
        'Inflation Upper': 0,
        'Label': 'Rising Growth & Falling Inflation'
    },
    {
        'Regime': 3,
        'S&P 500 Lower': float('-inf'),
        'S&P 500 Upper': 0,
        'Inflation Lower': 0,
        'Inflation Upper': float('inf'),
        'Label': 'Falling Growth & Rising Inflation'
    },
    {
        'Regime': 4,
        'S&P 500 Lower': float('-inf'),
        'S&P 500 Upper': 0,
        'Inflation Lower': float('-inf'),
        'Inflation Upper': 0,
        'Label': 'Falling Growth & Falling Inflation'
    }
]
