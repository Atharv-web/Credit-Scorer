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

# get the wallet features
def get_wallet_features(df: pd.DataFrame) -> pd.DataFrame:
    # convert to proper amount
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')

    df['usd_value'] = df['amount'] * df['assetPriceUSD']

    # convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # group by userWallet
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
        }

        features.append(row)
    return pd.DataFrame(features) 


def give_credit_scores(df):
    wallet_ids = df['userWallet']
    x = df.drop(['userWallet','last_tx','first_tx'],axis =1)
    x.fillna(x.mean(),inplace=True)
    model = IsolationForest(n_estimators=100,random_state= 42,contamination='auto')
    model.fit(x)
    raw_scores = model.decision_function(x)
    new_raw_scores = model.score_samples(x)
    inverted_scores = -new_raw_scores

    scaler = MinMaxScaler(feature_range=(0,1000))
    scores = scaler.fit_transform(inverted_scores.reshape(-1,1)).flatten()
    df['credit_score'] = scores

# function to plot the distribution graph
def vis_graph(df,image_file):
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
    plt.savefig(image_file)

# function to write the analysis.md file
def write_analysis(df,report_file,image_file):
    with open(report_file, "w") as f:
        f.write("# Analysis of Given wallets\n")
        f.write("## Score Distribution\n")
        f.write(f"![Distribution]({image_file})\n\n")
        f.write("## Summary\n")
        f.write(f"- Total wallets analyzed: {len(df)}\n")
        
        perfect_score_count = (df["credit_score"] == 1000).sum()
        zero_score_count = (df["credit_score"] == 0).sum()
        f.write(f"- Number of wallets with perfect credit score (1000): {perfect_score_count}\n\n")
        f.write(f"- Number of wallets with zero credit score (0):{zero_score_count}\n\n")

        top_wallets = df.sort_values(by="credit_score", ascending=False).head(3)
        f.write("## Top 3 Wallets by Credit Score\n")
        f.write("| Wallet | Credit Score | Net Borrow USD | Net Deposit USD | Asset Diversity | Liquidations |\n")
        f.write("|--------|--------------|----------------|-----------------|-----------------|--------------|\n")
        for _, row in top_wallets.iterrows():
            f.write(f"| {row['userWallet']} | {row['credit_score']} | {row['net_borrow_usd']:.2f} | {row['net_deposit_usd']:.2f} | {row['asset_diversity']:.2f} | {row['liquidation_count']} |\n")
        f.write("\n")

        bottom_wallets = df.sort_values(by="credit_score", ascending=True).head(3)
        f.write("## Bottom 3 Wallets by Credit Score\n")
        f.write("| Wallet | Credit Score | Total USD | Net Borrow USD | Net Deposit USD | Asset Diversity | Liquidations |\n")
        f.write("|--------|---------------|-----------|----------------|-----------------|-----------------|--------------|\n")
        for _, row in bottom_wallets.iterrows():
            f.write(f"| {row['userWallet']} | {row['credit_score']} | {row['total_usd']:.2f} | {row['net_borrow_usd']:.2f} | {row['net_deposit_usd']:.2f} | {row['asset_diversity']:.2f} | {row['liquidation_count']} |\n")
        f.write("\n")


INPUT_JSON_FILE = "user-wallet-transactions.json"
OUTPUT_CSV_FILE = "user-wallet-transactions.csv"
CSV_FEATURES_FILE = "wallet_features.csv"
DISTRIBUTION_GRAPH_FILE = "new.png"
ANALYSIS_FILE = "analysis.md"

def main():
    converter(INPUT_JSON_FILE,OUTPUT_CSV_FILE)
    original_df = pd.read_csv(OUTPUT_CSV_FILE)
    features_df = get_wallet_features(original_df) # get wallet features
    features_df.to_csv(CSV_FEATURES_FILE,index=False) # save to csv file

    give_credit_scores(features_df) # get the credit scores for the features    
    vis_graph(features_df,DISTRIBUTION_GRAPH_FILE) # gets the graph distribution and save it to a png
    write_analysis(features_df,ANALYSIS_FILE,DISTRIBUTION_GRAPH_FILE) # writes the markdown file with the analysis.

main()