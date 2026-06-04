# Data Preprocessing & Cleaning Report

This report provides a comparison of the dataset before and after the preprocessing pipeline steps (duplicate removal, missing value imputation, outlier handling, categorical encoding, and feature scaling).

## 1. General Cleaning Metrics
| Metric | Before | After | Change / Detail |
| --- | --- | --- | --- |
| **Dataset Shape** | (15000, 16) | (15000, 16) | Removed 0 duplicates |
| **Missing Values** | 0 | 0 | Imputed using Median/Most Frequent strategy |
| **Duplicate Rows** | 0 | 0 | Cleaned |

## 2. Outlier Detection Summary (IQR Method)
Extreme transactional attributes often correlate with fraudulent activity. To limit extreme variance without losing these patterns, variables were capped at their 99.5th percentile prior to scaling.

| Feature | Outliers Before (IQR) | Outliers After (IQR) | Upper Cap Limit (99.5th percentile) |
| --- | --- | --- | --- |
| age | 0 | 0 | N/A (No cap) |
| hour | 0 | 0 | N/A (No cap) |
| day_of_week | 0 | 0 | N/A (No cap) |
| amount | 1154 | 1154 | 417.26 |
| distance_from_home | 1327 | 1327 | 94.02 |
| transactions_per_user | 12 | 12 | N/A (No cap) |
| avg_transaction_amount | 607 | 607 | N/A (No cap) |
| time_since_last_tx | 719 | 719 | N/A (No cap) |
| tx_velocity_1h | 602 | 602 | N/A (No cap) |

## 3. Categorical Feature Encoding
Categorical elements were transformed into binary features using One-Hot Encoding:
- **Original features:** ['category', 'device', 'amount_category']
- **One-Hot Encoded features:** ['category_entertainment', 'category_gas', 'category_grocery', 'category_online_retail', 'category_services', 'category_travel', 'device_mobile', 'device_pos', 'device_web', 'amount_category_Critical', 'amount_category_High', 'amount_category_Low', 'amount_category_Medium']

## 4. Scaled Feature Verification (Train Split)
Summary statistics after Standard Scaling (should have mean ~ 0, std ~ 1):
```
                age          hour   day_of_week        amount  distance_from_home  transactions_per_user  avg_transaction_amount  time_since_last_tx  tx_velocity_1h
count  1.200000e+04  1.200000e+04  1.200000e+04  1.200000e+04        1.200000e+04           1.200000e+04            1.200000e+04        1.200000e+04    1.200000e+04
mean   1.669775e-16 -7.371881e-17 -6.750156e-17  9.695948e-17        1.113184e-16          -1.569115e-16            2.670456e-16        1.610564e-16    3.552714e-17
std    1.000042e+00  1.000042e+00  1.000042e+00  1.000042e+00        1.000042e+00           1.000042e+00            1.000042e+00        1.000042e+00    1.000042e+00
min   -1.948549e+00 -1.647872e+00 -1.584153e+00 -8.525343e-01       -6.941082e-01          -1.564663e+00           -2.218395e+00       -9.638467e-01   -2.011805e-01
25%   -7.174237e-01 -9.193217e-01 -1.075937e+00 -5.957734e-01       -5.390536e-01          -8.358880e-01           -5.625476e-01       -7.159560e-01   -2.011805e-01
50%   -1.018610e-01 -4.506086e-02 -5.950367e-02 -3.300375e-01       -3.386753e-01          -3.001860e-03           -1.262722e-01       -3.125569e-01   -2.011805e-01
75%    7.775143e-01  8.292000e-01  9.569292e-01  1.870419e-01        9.647151e-02           7.257735e-01            3.761371e-01        3.808222e-01   -2.011805e-01
max    2.975953e+00  1.703461e+00  1.465146e+00  5.881249e+00        6.780443e+00           3.328543e+00            1.840663e+01        7.599556e+00    9.733660e+00
```
