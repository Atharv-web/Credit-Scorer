import asyncio
import aiohttp
import pandas as pd
import os
from dotenv import load_dotenv
from utils import safe_fetch, extract_features
from scoring import ScoreWallets

load_dotenv()

API_KEY = os.getenv('api_key')

INPUT_XLSX = 'Copy of Wallet id.xlsx'
FEATURES_CSV = 'extracted_features.csv'
OUTPUT_CSV = 'wallet_credit_scores.csv'

async def main():
    wallets_df = pd.read_excel(INPUT_XLSX)
    wallets = wallets_df['wallet_id'].tolist()
    print(f"Number of wallets: {len(wallets)}")

    async with aiohttp.ClientSession() as session:
        tasks = [safe_fetch(session, w) for w in wallets]
        transactions = await asyncio.gather(*tasks)

    features = [extract_features(tx_list, w) for tx_list, w in zip(transactions, wallets)]
    features = [f for f in features if f is not None]  # Filter out None
    features_df = pd.DataFrame(features)
    features_df.to_csv(FEATURES_CSV, index=False)

    scores = ScoreWallets(features_df)
    min_score, max_score = scores.min(), scores.max()
    features_df['credit_score'] = ((scores - min_score) / (max_score - min_score + 1e-9)) * 1000
    features_df[['userWallet', 'credit_score']].to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    asyncio.run(main())
