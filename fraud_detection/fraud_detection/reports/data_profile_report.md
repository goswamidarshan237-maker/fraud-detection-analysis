# Data Profile Report

## Overview
Source file: `data/raw/transactions.csv`
Dimensions: 15000 rows, 8 columns

## Features and Data Types
| Feature | Data Type | Missing Values |
| --- | --- | --- |
| transaction_id | str | 0 |
| age | int64 | 0 |
| hour | int64 | 0 |
| amount | float64 | 0 |
| distance_from_home | float64 | 0 |
| category | str | 0 |
| device | str | 0 |
| is_fraud | int64 | 0 |

## Summary Statistics
```
       transaction_id           age          hour        amount  distance_from_home category device      is_fraud
count           15000  15000.000000  15000.000000  15000.000000        15000.000000    15000  15000  15000.000000
unique          15000           NaN           NaN           NaN                 NaN        6      3           NaN
top          TX000000           NaN           NaN           NaN                 NaN  grocery    pos           NaN
freq                1           NaN           NaN           NaN                 NaN     5241   7472           NaN
mean              NaN     40.677000     11.593933     54.119306            9.163117      NaN    NaN      0.030267
std               NaN     11.692917      6.889050     71.173030           14.629030      NaN    NaN      0.171326
min               NaN     18.000000      0.000000      0.500000            0.050000      NaN    NaN      0.000000
25%               NaN     32.000000      6.000000     16.680000            1.990000      NaN    NaN      0.000000
50%               NaN     41.000000     12.000000     32.965000            4.510000      NaN    NaN      0.000000
75%               NaN     49.000000     18.000000     64.592500           10.162500      NaN    NaN      0.000000
max               NaN     88.000000     23.000000   2437.820000          288.430000      NaN    NaN      1.000000
```
