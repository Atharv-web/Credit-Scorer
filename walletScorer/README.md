# walletScorer

## Overview

The `walletScorer` module is a part of the Credit-Scorer project, designed to assess the creditworthiness of wallets. It automates the process of collecting wallet data, engineering relevant features, and assigning a credit score in a scalable and transparent manner. Since there was no specification of **chainName**, we are going to take Ethereum wallets data thereby **chainName** = `eth-mainnet`.

---

## Methodology

### Data Collection

- **Source:** The system collects wallet transaction data from the CovalentHQ API, which provides detailed, up-to-date blockchain information.
- **Method:** It uses asynchronous HTTP requests (`aiohttp`/`asyncio`) to efficiently fetch transaction histories for many wallets in parallel. To prevent api rate limits during api calls, the transaction data for every wallet is upto 20 transactions. If more transaction data is needed, you can change **page_size** and **page_number** which indicate the number of transactions and the page number. Default choice is **page_size** be 20 and **page_number** be 0 (first page only.)
- **Input:** Wallet IDs are taken from an Excel file, making it simple to batch-process large datasets.

### Feature Selection

- **Rationale:** Features are chosen to capture key aspects of financial behavior and wallet activity that correlate with creditworthiness:
    - **Transaction Activity:** Total transactions, unique counterparties, and contract interactions show engagement and network breadth.
    - **Financial Flow:** Total ETH sent/received, average transaction value, and wallet age provide insight into the wallet’s financial health.
    - **Reliability:** Number of failed transactions and active days help understand operational consistency and risk.
- **Scalability:** Feature extraction is designed to be robust against missing or incomplete data and can be extended to incorporate new features as the scoring model evolves.

### Scoring Method

- **Weighted Formula:** Each feature is assigned a weight reflecting its importance in assessing credit risk (e.g., more weight to transaction volume, less to failed transactions).
- **Computation:** A raw score is computed as a weighted sum of all features, which is then normalized to a standard scale (0–1000) for easy comparison across wallets.
- **Transparency & Extensibility:** The scoring logic is implemented in a modular and transparent way, enabling easy updates or integration of machine learning models in the future.

---

## Usage

1. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare Environment:**
   - Get your CovalentHQ API key from GoldRush platform and save it in `.env` file.
   - List wallet IDs in `Copy of Wallet id.xlsx` (column: `wallet_id`).
3. **Run the Scorer:**
   ```sh
   python main.py
   ```
   Outputs: `extracted_features.csv` (features taken from data), `wallet_credit_scores.csv` (csv file that contains the final scores with respective wallet ids).

---

## File Structure

- `main.py` — Orchestrates reading, processing, and output.
- `scoring.py` — Contains feature weighting and scoring logic.
- `utils.py` — Provides async data fetching and feature extraction utilities.
- `requirements.txt` — Python dependencies.

---