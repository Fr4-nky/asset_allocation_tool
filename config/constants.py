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
}

REGIME_BG_ALPHA = 0.13  # Use the same alpha for all charts and tables
regime_bg_colors = {
    1: f'rgba(0, 128, 255, {REGIME_BG_ALPHA})',   # Rising Growth, Rising Inflation - Light Blue
    2: f'rgba(0, 255, 0, {REGIME_BG_ALPHA})',     # Rising Growth, Falling Inflation - Light Green
    3: f'rgba(255, 0, 0, {REGIME_BG_ALPHA})',     # Falling Growth, Rising Inflation - Light Red
    4: f'rgba(255, 255, 0, {REGIME_BG_ALPHA})',   # Falling Growth, Falling Inflation - Light Yellow
    'Unknown': 'lightgrey'
}

regime_legend_colors = {
    1: 'rgba(0, 128, 255, 0.7)',   # Light Blue, more visible
    2: 'rgba(0, 255, 0, 0.7)',    # Light Green
    3: 'rgba(255, 0, 0, 0.7)',    # Light Red
    4: 'rgba(255, 255, 0, 0.7)',  # Light Yellow
    'Unknown': 'lightgrey'
}

regime_labels_dict = {
    1: 'Rising Growth & Rising Inflation',
    2: 'Rising Growth & Falling Inflation',
    3: 'Falling Growth & Rising Inflation',
    4: 'Falling Growth & Falling Inflation',
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
