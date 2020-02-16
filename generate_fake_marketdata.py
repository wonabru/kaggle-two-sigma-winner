"""
simple script for generating fake marketdata.
"""
import pandas as pd
import numpy as np
import datetime as dt


marketdata = pd.read_csv("./data/marketdata_sample.csv")

submission_file = pd.read_csv("./data/submission_sample.csv")

marketdata['time'] = pd.to_datetime(marketdata['time'])

def fake_price(price):
    return price * np.exp(np.random.randn()/100)


def fake_return(ret):
    return np.random.randn()/100 if np.random.rand() > 0.99 else np.random.randn()


def fake_return10(ret):
    return np.random.randn()/100 * np.sqrt(10) if np.random.rand() > 0.99 else np.random.randn() * np.sqrt(10)


def fake_row(row):
    next_time = row['time'] + dt.timedelta(days=1)
    
    if next_time.weekday() == 5:
        next_time += dt.timedelta(days=2)
    if next_time.weekday() == 6:
        next_time += dt.timedelta(days=1)
    
    next_row = [
        next_time,
        row['assetCode'],
        row['assetName'],
        np.random.randint(0, 2),
        round(row['volume'] * np.exp(np.random.randn() / 10)),
        fake_price(row['close']),
        fake_price(row['close']),
        fake_return(row['returnsClosePrevRaw1']),
        fake_return(row['returnsOpenPrevRaw1']),
        fake_return(row['returnsClosePrevMktres1']),
        fake_return(row['returnsOpenPrevMktres1']),
        fake_return10(row['returnsClosePrevRaw10']),
        fake_return10(row['returnsOpenPrevRaw10']),
        fake_return10(row['returnsClosePrevMktres10']),
        fake_return10(row['returnsOpenPrevMktres10']),
        fake_return10(row['returnsOpenNextMktres10'])
    ]
    return np.array(next_row)

marketdata_tmp = pd.read_csv("./data/fake_marketdata_beginning.csv")
marketdata_tmp['time'] = pd.to_datetime(marketdata_tmp['time'])

fake_marketdata = pd.DataFrame(columns=marketdata.columns)
while True:
    fake_marketdata_tmp = pd.DataFrame(columns=marketdata.columns)
    for index, row in marketdata_tmp.iterrows():
        next_row = fake_row(row)
        next_row_df = pd.DataFrame(data=[next_row], columns=marketdata_tmp.columns)
        fake_marketdata_tmp = fake_marketdata_tmp.append(next_row_df)

    marketdata_tmp = fake_marketdata_tmp.copy()

    fake_marketdata = fake_marketdata.append(fake_marketdata_tmp)
    if pd.to_datetime(fake_marketdata['time'].values[-1]) >= pd.to_datetime("2019-01-08"):
        break

fake_marketdata.to_csv("./data/fake_marketdata.csv", index=False)