import json
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def convert_to_csv(transaction):
    data = {
        "_id": transaction["_id"].get("$oid"),
        "userWallet": transaction["userWallet"],
        "network": transaction["network"],
        "protocol": transaction["protocol"],
        "txHash": transaction["txHash"],
        "logId": transaction["logId"],
        "timestamp": transaction["timestamp"],
        "blockNumber": transaction["blockNumber"],
        "action": transaction["action"],
        "type": transaction["actionData"].get("type"),
        "amount": transaction["actionData"].get("amount"),
        "assetSymbol": transaction["actionData"].get("assetSymbol"),
        "assetPriceUSD": transaction["actionData"].get("assetPriceUSD"),
        "poolId": transaction["actionData"].get("poolId"),
        "userId": transaction["actionData"].get("userId"),
        "createdAt": transaction["createdAt"]["$date"],
        "updatedAt": transaction["updatedAt"]["$date"]
    }
    return data

def converter(json_file:str,csv_file:str):
    with open(json_file,"r") as j:
        data = json.load(j)

    tx_history = [convert_to_csv(tx) for tx in data]
    with open(csv_file, "w") as c:
        writer = csv.DictWriter(c,fieldnames=tx_history[0].keys())
        writer.writeheader()
        writer.writerows(tx_history)
    print(f"converted json file to CSV for Machine Learning, file name is {csv_file}")

# convert the raw csv file to a meaningful one.
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

# fucntion to generate credit scores for the dataset.
def give_credit_scores(df):
    wallet_ids = df['userWallet'] # keeping wallet ids.
    x = df.drop(['userWallet','last_tx','first_tx'],axis = 1)
    x.fillna(x.mean(),inplace=True)
    model = IsolationForest(n_estimators=100,random_state=42, contamination='auto')
    model.fit(x)
    raw_scores = model.decision_function(x)
    scaler = MinMaxScaler(feature_range=(0,1000))
    scores = scaler.fit_transform(raw_scores.reshape(-1,1)).flatten()
    df['credit_score'] = scores

# function to analyse the wallets after getting the credits
def analyze_features(df):
    bins = list(range(0,1100,100))
    labels = [f"{b}-{b+99}" for b in bins[:-1]]
    df['score_range'] = pd.cut(df['credit_score'], bins=bins,labels=labels,include_lowest=True)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='score_range', order=labels, palette='coolwarm')
    plt.title('Distribution of Wallet Credit Scores')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("wallet_score_distribution.png")

INPUT_JSON_FILE = "user-wallet-transactions.json"
OUTPUT_CSV_FILE = "user-wallet-transactions.csv"
CSV_FEATURES_FILE = "wallet_features.csv"
ANALYSIS_FILE = "analysis.md"

def main():
    converter(INPUT_JSON_FILE,OUTPUT_CSV_FILE)
    original_df = pd.read_csv(OUTPUT_CSV_FILE)
    
    features_df = get_wallet_features(original_df) # get wallet features
    features_df.to_csv(CSV_FEATURES_FILE,index=False) # convert them into a csv file
    
    give_credit_scores(features_df) # get the credit scores for the features
    
    analyze_features(features_df)

main()