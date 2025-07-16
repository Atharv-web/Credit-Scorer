# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import MinMaxScaler
# # from ace_tools import display_dataframe_to_user

# # Load feature data
# df = pd.read_csv('wallet_features.csv')

# # Preserve wallet identifiers
# wallet_ids = df['userWallet']

# # Drop non-numeric columns that are not useful for modeling
# drop_cols = ['userWallet', 'first_tx', 'last_tx']
# feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# # Handle missing values by imputing column means
# feature_df.fillna(feature_df.mean(), inplace=True)

# # Fit Isolation Forest for anomaly detection
# model = IsolationForest(n_estimators=100, random_state=42, contamination='auto')
# model.fit(feature_df)
# raw_scores = model.decision_function(feature_df)  # Higher = more "normal"

# # Scale to [0, 1000]
# scaler = MinMaxScaler(feature_range=(0, 1000))
# credit_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

# # Add scores back to DataFrame
# df['credit_score'] = credit_scores

# # Display top 10 highest-scoring wallets
# # top_10 = df[['userWallet', 'credit_score']].sort_values(by='credit_score', ascending=False).head(10)
# # display_dataframe_to_user("Top 10 Wallet Credit Scores", top_10)

# # Display bottom 10 lowest-scoring wallets
# # bottom_10 = df[['userWallet', 'credit_score']].sort_values(by='credit_score', ascending=True).head(10)
# # display_dataframe_to_user("Bottom 10 Wallet Credit Scores", bottom_10)

# # Save to CSV
# # df[['userWallet', 'credit_score']].to_csv('/mnt/data/wallet_credit_scores.csv', index=False)



# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import MinMaxScaler


# def generate_wallet_credit_scores(csv_path: str, output_path: str = "wallet_credit_scores.csv"):
#     # Load dataset
#     df = pd.read_csv(csv_path)

#     # Preserve wallet identifiers
#     wallet_ids = df['userWallet'] if 'userWallet' in df.columns else df.index

#     # Drop non-numeric or unnecessary columns
#     drop_cols = ['userWallet', 'first_tx', 'last_tx']
#     features = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

#     # Fill missing values
#     features.fillna(features.mean(), inplace=True)

#     # Fit Isolation Forest model
#     iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
#     iso_forest.fit(features)

#     # Get anomaly scores (the higher, the more normal)
#     raw_scores = iso_forest.decision_function(features)

#     # Normalize scores to 0-1000
#     scaler = MinMaxScaler(feature_range=(0, 1000))
#     normalized_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

#     # Add scores to dataframe
#     df['credit_score'] = normalized_scores

#     # Save results
#     df[['userWallet', 'credit_score']].to_csv(output_path, index=False)
#     print(f"Credit scores saved to {output_path}")

#     # Optionally return the top and bottom scoring wallets
#     top_10 = df[['userWallet', 'credit_score']].sort_values(by='credit_score', ascending=False).head(10)
#     bottom_10 = df[['userWallet', 'credit_score']].sort_values(by='credit_score', ascending=True).head(10)
    
#     return top_10, bottom_10


# # Example usage
# if __name__ == "__main__":
#     top, bottom = generate_wallet_credit_scores("/mnt/data/wallet_features.csv")
#     print("Top 10 Credit Scores:")
#     print(top)
#     print("\nBottom 10 Credit Scores:")
#     print(bottom)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Load feature dataset
df = pd.read_csv("wallet_features.csv")

# Preserve wallet IDs
wallet_ids = df['userWallet']

# Drop irrelevant columns
drop_cols = ['userWallet', 'first_tx', 'last_tx']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Fill missing values
X.fillna(X.mean(), inplace=True)

# Fit Isolation Forest model
model = IsolationForest(n_estimators=100, random_state=42, contamination='auto')
model.fit(X)
raw_scores = model.decision_function(X)  # Higher = more normal

# Scale scores to 0-1000
scaler = MinMaxScaler(feature_range=(0, 1000))
scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
df['credit_score'] = scores

# --- Analysis ---

# Define score buckets
bins = list(range(0, 1100, 100))
labels = [f"{b}-{b+99}" for b in bins[:-1]]
df['score_range'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)

# Plot score distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='score_range', order=labels, palette='coolwarm')
plt.title('Distribution of Wallet Credit Scores')
plt.xlabel('Score Range')
plt.ylabel('Number of Wallets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("wallet_score_distribution.png")
plt.show()

# Analyze low scoring wallets (< 200)
low_scoring = df[df['credit_score'] < 200]
print("\n--- Low Scoring Wallets (<200) ---")
print("Count:", len(low_scoring))
print("Common actions:", low_scoring['action'].value_counts().head())
print("Assets used:", low_scoring['assetSymbol'].value_counts().head())

# Analyze high scoring wallets (> 800)
high_scoring = df[df['credit_score'] > 800]
print("\n--- High Scoring Wallets (>800) ---")
print("Count:", len(high_scoring))
print("Common actions:", high_scoring['action'].value_counts().head())
print("Assets used:", high_scoring['assetSymbol'].value_counts().head())

# Save results
df[['userWallet', 'credit_score', 'score_range']].to_csv("wallet_credit_scores.csv", index=False)
