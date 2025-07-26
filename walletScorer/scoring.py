def ScoreWallets(df):
    score = (
        0.13 * df['total_transactions'] -
        0.17 * df['num_failed'] +
        0.12 * df['avg_tx_value'] +
        0.20 * df['total_eth_sent'] +
        0.20 * df['total_eth_received'] +
        0.15 * df['wallet_age_days'] +
        0.15 * df['active_days'] +
        0.18 * df['unique_counterparties']
    )
    return score
