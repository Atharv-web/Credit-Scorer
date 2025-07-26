# walletScorer

## Overview

The `walletScorer` module is a core part of the Credit-Scorer project, designed to assess the creditworthiness of Ethereum wallets. It automates the process of collecting wallet data, engineering relevant features, and assigning a credit score in a scalable and transparent manner.

---

## Methodology

### Data Collection

- **Source:** The system collects wallet transaction data from the CovalentHQ API, which provides detailed, up-to-date blockchain information.
- **Method:** It uses asynchronous HTTP requests (`aiohttp`/`asyncio`) to efficiently fetch transaction histories for many wallets in parallel. This approach supports scalability to thousands of wallets without significant performance bottlenecks.
- **Input:** Wallet IDs are ingested from an Excel file, making it simple to batch-process large datasets.

### Feature Selection

- **Rationale:** Features are chosen to capture key aspects of financial behavior and wallet activity that correlate with creditworthiness and trust:
    - **Transaction Activity:** Total transactions, unique counterparties, and contract interactions signal engagement and network breadth.
    - **Financial Flow:** Total ETH sent/received, average transaction value, and wallet age provide insight into the wallet’s financial health.
    - **Reliability:** Number of failed transactions and active days help gauge operational consistency and risk.
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
   - Set your CovalentHQ API key in a `.env` file.
   - List wallet IDs in `Copy of Wallet id.xlsx` (column: `wallet_id`).
3. **Run the Scorer:**
   ```sh
   python main.py
   ```
   Outputs: `extracted_features.csv` (all features), `wallet_credit_scores.csv` (final scores).

---

## File Structure

- `main.py` — Orchestrates reading, processing, and output.
- `scoring.py` — Contains feature weighting and scoring logic.
- `utils.py` — Provides async data fetching and feature extraction utilities.
- `requirements.txt` — Python dependencies.

---

## License

See the main repository for license information.

**Author:** [Atharv-web](https://github.com/Atharv-web)