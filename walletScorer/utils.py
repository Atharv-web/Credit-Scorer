import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

API_KEY = None
CHAIN= "eth-mainnet"
PAGE_SIZE=20
PAGE_NUMBER = 0

async def fetch_transactions(session, wallet):
    url = f"https://api.covalenthq.com/v1/{CHAIN}/address/{wallet}/transactions_v2/"
    params = {"page-size": PAGE_SIZE, "page-number": PAGE_NUMBER}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get('data', {}).get('items', [])
    except aiohttp.ClientError as e:
        print(f"Error fetching data for wallet: {wallet}, Error: {e}")
        return []

semaphore = asyncio.Semaphore(5)

async def safe_fetch(session, wallet):
    async with semaphore:
        return await fetch_transactions(session, wallet)
    
def extract_features(tx_list, wallet):
    df = pd.DataFrame(tx_list)
    if df.empty:
        return None
    # normalize columns
    df['timestamp'] = pd.to_datetime(df['block_signed_at'],format='%Y-%m-%dT%H:%M:%SZ')
    df['value_eth'] = pd.to_numeric(df['value'], errors='coerce') / 1e18
    df['success'] = df['successful'].astype(bool)
    df['outgoing'] = df['from_address'].str.lower() == wallet.lower()
    df['has_contract'] = df['log_events'].apply(lambda x: len(x) > 0)

    features = {
        'userWallet': wallet,
        'total_transactions': len(df),
        'num_outgoing': int(df['outgoing'].sum()),
        'num_incoming': int((~df['outgoing']).sum()),
        'num_failed': int((~df['success']).sum()),
        'avg_tx_value': float(df['value_eth'].mean()),
        'total_eth_sent': float(df[df['outgoing']]['value_eth'].sum()),
        'total_eth_received': float(df[~df['outgoing']]['value_eth'].sum()),
        'avg_gas_price': float(df['gas_price'].astype(float).mean()),
        'total_gas_spent': float(df['gas_spent'].astype(float).sum()),
        'active_days': int(df['timestamp'].dt.date.nunique()),
        'wallet_age_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        'contract_interaction_ratio': float(df['has_contract'].mean()),
        'unique_counterparties': int(pd.concat([df['from_address'], df['to_address']]).nunique())
    }
    return features
