# Credit-Scorer

Credit-Scorer is a Python toolkit for analyzing the creditworthiness of blockchain wallets using transaction data. This repository provides two main approaches for scoring: a transparent rule-based method and a machine learning-based method (see the `MLScorer` folder for details).
Since the rule based method gives more reasoning for evaluating the scores, while the ML approach gives a comprehensive understanding of the dataset due to variations in activity of the wallets. 
I implemented both approaches.

---

## Scoring Logic & Transparency


### (Same for Rule based and ML based)

- **Feature Extraction:** For each wallet, the following features are computed: 
  - `tx_per_day`: Average transactions per day
  - `duration_days`: Total active days
  - `net_deposit_usd`: Deposits minus withdrawals (USD)
  - `net_borrow_usd`: Borrows minus repayments (USD)
  - `avg_usd`: Average transaction value (USD)
  - `asset_diversity`: Number of unique assets interacted with
  - `growth_ratio`: Growth in USD activity (recent/early)
  - `liquidation_count`: Number of liquidations (penalty)
  - `active_days`: Number of days with activity
  - `deposit_usd`: Total deposited USD
  - `repay_usd`: Total repaid USD

### Rule Based Scoring 

- **Scoring Formula:**  
  Features are scaled [0, 1]. A weighted sum is applied:
  ```
  raw_score = sum(feature_i * weight_i)
  ```
  Weights are documented in `CreditScorer.py`.

- **Normalization:**  
  Raw scores are scaled to a [0, 1000] credit score for transparency.

- **Extensibility:**  
  - Add features in `get_wallet_features`.
  - Adjust weights in the `weights` dictionary.
  - Analysis report and score distribution are generated for auditability.

---

### Machine Learning Scoring (`MLScorer/MLCreditScorer.py`)

- **Model:**  
  Uses an unsupervised Isolation Forest to model typical wallet behavior and assign scores.
  Outlier wallets are scored lower; typical behavior is scored higher. Uses a tree based approach.
  The ML approach provides inference based on anomalous activity which takes factors like asset diversity, liquidations, net borrow, net deposit, growth ratio etc.
  Higher activity in wallets like high deposit/borrowing transactions, or higher asset diversity , minimum liquidations etc. makes features have shorter length.
  The model considers this to be anomalous behaviour and hence gets a lower score.
  Wallets with lower activity were given high scores as they could be easliy isolated in the tree, therfore were less anomalous and were more normal or modelled typical behaviour. This gave these wallets a higher score.
  Inverting the logic and the scores provides a approach that helps in generating scores to based on transaction activity in the correct manner.

- **Scoring:**  
  - Model outputs anomaly scores (lower = more anomalous).
  - Scores are inverted and normalized to [0, 1000].
  - This approach is extensible: just add more features to `get_wallet_features` or tune the ML model.

- **Transparency:**  
  Although ML-based, the feature pipeline and scoring steps are documented in code and analysis files (`MLScorer/analysis.md`).

---

## How to Use

### Prerequisites
- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

Install dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Rule-Based Scorer
1. Place your wallet transaction JSON in the root directory (named `user-wallet-transactions.json`).
2. Run:
    ```bash
    python CreditScorer.py
    ```
3. Outputs:
    - `wallet_features.csv`: Extracted features (from feature engineering)
    - `analysis.md`: Analysis and behaviour of wallets
    - `credit_score_distribution.png`: Score distribution

### ML-Based Scorer
1. Place your wallet transaction JSON in the `MLScorer` folder (named `user-wallet-transactions.json`).
2. Run:
    ```bash
    python MLScorer/MLCreditScorer.py
    ```
3. Outputs:
    - `wallet_features.csv`: Extracted features (from feature engineering)
    - `analysis.md`: Analysis and behaviour of wallets
    - `new.png`: Score distribution


---

## Folder: MLScorer

Contains a ML-based scoring pipeline.
- `MLCreditScorer.py`: ML logic (Isolation Forest)
- `analysis.md`: ML scoring analysis
- Output files: features and distribution graphs

---
