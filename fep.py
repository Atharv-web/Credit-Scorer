import pandas as pd
import numpy as np

# This is the feature engineering pipeline

def get_wallet_features(df: pd.DataFrame) -> pd.DataFrame:
    token_decimals = {
        'USDC': 6,
        'USDT': 6,
        'DAI': 18,
        'WETH': 18,
        'WBTC': 8,
        'WMATIC': 18,
        'AAVE': 18,
        'WPOL': 18
    }

    # 2. Convert to proper amount
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')

    def convert_amount(row):
        symbol = row['assetSymbol']
        decimals = token_decimals.get(symbol, 18) # if nan value found, then it gets replaced by 18(default value)
        return row['amount'] / (10 ** decimals)

    df['amount_converted'] = df.apply(convert_amount, axis=1)
    df['usd_value'] = df['amount_converted'] * df['assetPriceUSD']

    # 3. Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # 4. Group by userWallet
    grouped = df.groupby('userWallet')

    features = []

    for wallet, group in grouped:
        actions = group['action'].value_counts(normalize=True).to_dict()
        first_time = group['timestamp'].min()
        last_time = group['timestamp'].max()
        duration_days = (last_time - first_time).days or 1

        deposit_usd = group.loc[group['action'].str.lower() == 'deposit', 'usd_value'].sum()
        redeem_usd = group.loc[group['action'].str.lower() == 'redeemunderlying', 'usd_value'].sum()
        borrow_usd = group.loc[group['action'].str.lower() == 'borrow', 'usd_value'].sum()
        repay_usd = group.loc[group['action'].str.lower() == 'repay', 'usd_value'].sum()
        liquidation_count = group.loc[group['action'].str.lower() == 'liquidationcall'].shape[0]

        active_days = group['timestamp'].dt.date.nunique()
        asset_diversity = group['assetSymbol'].nunique()
        tx_per_day = group.shape[0] / duration_days

        early_window = first_time + pd.Timedelta(days=duration_days * 0.3)
        recent_window = last_time - pd.Timedelta(days=duration_days * 0.3)

        early_usd = group.loc[group['timestamp'] < early_window, 'usd_value'].sum()
        recent_usd = group.loc[group['timestamp'] > recent_window, 'usd_value'].sum()
        growth_ratio = recent_usd / early_usd if early_usd > 0 else np.nan

        row = {
            'userWallet': wallet,
            'tx_count': group.shape[0],
            'tx_per_day': tx_per_day,
            'active_days': active_days,
            'first_tx': first_time,
            'last_tx': last_time,
            'duration_days': duration_days,
            'total_usd': group['usd_value'].sum(),
            'avg_usd': group['usd_value'].mean(),
            'deposit_usd': deposit_usd,
            'redeem_usd': redeem_usd,
            'borrow_usd': borrow_usd,
            'repay_usd': repay_usd,
            'net_deposit_usd': deposit_usd - redeem_usd,
            'net_borrow_usd': borrow_usd - repay_usd,
            'asset_diversity': asset_diversity,
            'growth_ratio': growth_ratio,
            'liquidation_count': liquidation_count,
            # Add action ratios - deposit/borrowing/redeemunderlying
            **{f"action_ratio_{k}": v for k, v in actions.items()}
        }

        features.append(row)

    return pd.DataFrame(features)

df =pd.read_csv('user-wallet-transactions.csv')
features_df = get_wallet_features(df)
features_df.to_csv("wallet_features.csv",index=False)