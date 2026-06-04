# Feature Engineering Report

This report documents the engineered features designed to improve model sensitivity to credit card fraud, along with the domain-specific rationale for each feature.

## 1. Engineered Features & Descriptions

### A. Transaction Amount Category (`amount_category`)
- **Definition:** Categorizes transaction amounts into risk bins (`Low` < $20, `Medium` $20-$100, `High` $100-$500, `Critical` >= $500).
- **Why it helps:** Fraud transactions typically feature abnormally high prices (unusual purchases of high-value commodities). Risk binning allows classifiers to learn non-linear thresholds on value spikes directly.

### B. Cumulative Transactions per User (`transactions_per_user`)
- **Definition:** A running count of the total number of transactions recorded for each user.
- **Why it helps:** Established accounts (high transaction counts) show stable behavior patterns. A new card/account running multiple heavy transactions quickly (low cumulative counts but high frequency) is a classic indicator of synthetic profile fraud.

### C. Average Transaction Amount (`avg_transaction_amount`)
- **Definition:** The running average amount of all historical transactions made by the user.
- **Why it helps:** Fraudsters usually spend way more than a victim's normal transactional footprint. Comparing the current amount to the user's running historical average flags standard deviation spikes.

### D. Time Since Previous Transaction (`time_since_last_tx`)
- **Definition:** Time delta (in minutes) between the current transaction and the user's immediately preceding transaction. First transactions default to `-1`.
- **Why it helps:** Physically impossible velocity (e.g. card run in two locations within 2 minutes) or rapid repeated card testing (multiple attempts within seconds) yields tiny time intervals. A low delta is highly correlated with script-driven automated fraud.

### E. Rolling Transaction Velocity (`tx_velocity_1h`)
- **Definition:** Rolling count of transactions made by the user within the preceding 1 hour.
- **Why it helps:** Legitimate cardholders rarely execute multiple transactions in an hour. Multiple hits in a short window imply card testing or rapid asset draining.

## 2. Sample Data with Engineered Features
```
user_id  amount amount_category  transactions_per_user  avg_transaction_amount  time_since_last_tx  tx_velocity_1h
 USR000   53.90          Medium                      1                   53.90               -1.00               0
 USR000    7.33             Low                      2                   30.62              179.20               0
 USR000    8.52             Low                      3                   23.25             2456.10               0
 USR000   22.88          Medium                      4                   23.16              269.13               0
 USR000  206.15            High                      5                   59.76             1255.07               0
 USR000   14.13             Low                      6                   52.15              913.47               0
 USR000   26.89          Medium                      7                   48.54             2679.77               0
 USR000   19.37             Low                      8                   44.90             1271.07               0
 USR000   19.03             Low                      9                   42.02              344.67               0
 USR000   39.58          Medium                     10                   41.78             1447.40               0
 USR000  134.39            High                     11                   50.20              541.98               0
 USR000   52.25          Medium                     12                   50.37              506.33               0
 USR000   14.83             Low                     13                   47.63             1710.35               0
 USR000  519.69        Critical                     14                   81.35             1326.08               0
 USR000   35.82          Medium                     15                   78.32              538.35               0
```
