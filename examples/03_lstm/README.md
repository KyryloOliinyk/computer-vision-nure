# Time Series Forecasting with LSTM Networks

**Course:** Deep Learning in Computer Vision Technologies  
**Department:** IST, NURE (Kharkiv National University of Radio Electronics)

---

## Variants Overview

| # | Dataset                     | 
|---|-----------------------------|
| 1 | Bitcoin BTC/USD Price       |
| 2 | Gold price                  | 
| 3 | Apple Stock                 |
| 4 | Airline Passengers          | 
| 5 | Household Power Consumption | 
| 6 | Shampoo Sales               |
| 7 | Monthly Sunspot Number      |

---

## Variant 1 — Bitcoin Price

| Field                  | Value                                                           |
|------------------------|-----------------------------------------------------------------|
| **Dataset**            | Daily Bitcoin (BTC/USD) closing price                           |
| **Source**             | Yahoo Finance                                                   |
| **Task**               | Forecast closing price for the next 7 days                      |
| **Train / Test split** | 80 / 20                                                         |
| **Extra task**         | Study the effect of Dropout rate (0.1, 0.2, 0.3) on overfitting |

**Download:**
```python
# Option A — via yfinance  (pip install yfinance)
import yfinance as yf

df = yf.download('BTC-USD', start='2018-01-01', end='2022-12-31')
df = df[['Close']].rename(columns={'Close': 'Price'})

```

---

## Variant 4 — International Airline Passengers

| Field        | Value                                            |
|--------------|--------------------------------------------------|
| **Dataset**  | International airline passengers                 |
| **Source**   | Kaggle / Seaborn datasets                        |
| **Task**     | Forecast passenger count for the next 12 months  |

**Download:**
```python
import pandas as pd

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(URL, parse_dates=['Month'], index_col='Month')
```

---

## Variant 5 — Household Power Consumption

| Field                  | Value                                                                                 |
|------------------------|---------------------------------------------------------------------------------------|
| **Dataset**            | Individual household electric power consumption                                       |
| **Source**             | UCI ML Repository                                                                     |
| **Task**               | Forecast daily power consumption for the next 7 days                                  |
| **Extra task**         | Study the effect of epoch count (50, 100, 150, 200) on accuracy; plot learning curves |

**Download:**
```python
import pandas as pd
# https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

df = pd.read_csv('household_power_consumption.txt', sep=';',
                 parse_dates={'datetime': ['Date', 'Time']},
                 infer_datetime_format=True,
                 low_memory=False,
                 na_values=['?'])

# Aggregate to daily resolution
df_daily = df.resample('D', on='datetime')['Global_active_power'].sum().dropna()
df_daily = df_daily.to_frame(name='Power')
```

---

## Variant 6 — Shampoo Sales

| Field                  | Value                                                                       |
|------------------------|-----------------------------------------------------------------------------|
| **Dataset**            | Shampoo sales over three years                                              |
| **Source**             | Jason Brownlee / Kaggle                                                     |
| **Task**               | Forecast sales volume for the next 6 months                                 |
| **Train / Test split** | 80 / 20                                                                     |

**Download:**
```python
import pandas as pd

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
df = pd.read_csv(URL)
df.columns = ['Month', 'Sales']
```

---

## Variant 7 — Monthly Sunspot Number

| Field                  | Value                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------|
| **Dataset**            | Monthly mean total sunspot number                                                             |
| **Source**             | SILSO — Royal Observatory of Belgium                                                          |
| **Extra task**         | Find the optimal lookback window (12, 24, 36 months); analyse cyclic patterns learned by LSTM |

**Download:**
```python
import pandas as pd

URL = 'https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv'

df = pd.read_csv(URL, sep=';', header=None,
                 names=['year', 'month', 'date_frac', 'spots',
                        'std', 'n_obs', 'definitive'])

df = df[df['year'] >= 1900].copy()
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.set_index('date')[['spots']]
df = df[df['spots'] >= 0]   # remove missing values coded as -1
```

---
