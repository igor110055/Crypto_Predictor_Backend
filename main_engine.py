import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import requests
import time
import pickle
from itertools import product
import sys
import threading
import math
import random
import statistics
import hmac
import hashlib
from datetime import datetime, timedelta
import datetime as dt
import ta

base_api = "https://api.binance.com"
all_data = {}

# General Functions
def get_symbol_interval(symbol):
    """Given a 'symbol' eg -> 'BTCUSDT1h'
    it returns -> 'BTCUSDT', '1h'
    """
    if "30m" in symbol:
        return symbol.replace("30m", ""), "30m"
    elif "1h" in symbol:
        return symbol.replace("1h", ""), "1h"
    elif "2h" in symbol:
        return symbol.replace("2h", ""), "2h"
    elif "4h" in symbol:
        return symbol.replace("4h", ""), "4h"
    elif "6h" in symbol:
        return symbol.replace("6h", ""), "6h"
    elif "12h" in symbol:
        return symbol.replace("12h", ""), "12h"
    elif "1D" in symbol:
        return symbol.replace("1D", ""), "1D"
    else:
        return symbol, "1h"


# Main Engine for Indicators used. To be replaced with ta library: future work
class Fin_Indicator:
    """Class definition for various indicators used in this project."""

    def __init__(self, main_dataset):
        self.dataset = main_dataset.copy()
        self.open_ = dataset["open"]
        self.high = dataset["high"]
        self.low = dataset["low"]
        self.close = dataset["close"]
        self.volume = dataset["volume"]

    def weighted_close_fxn(self):
        """Returns weighted close values of a given dataset"""
        weighted_close = (self.high + self.low + (self.close * 2)) / 4
        return weighted_close

    def bbands(self, typ="TP", timeperiod=24, num_deviations=2):
        """Returns Bollinger Bands values of a given dataset"""
        if typ == "TP":
            typical_price = (self.high + self.low + self.close) / 3
        elif typ == "weighted_close":
            typical_price = self.weighted_close_fxn()
        middleband = typical_price.rolling(window=timeperiod).mean()
        stdeviation = typical_price.rolling(window=timeperiod).std()
        upperband = middleband + (num_deviations * stdeviation)
        lowerband = middleband - (num_deviations * stdeviation)
        return upperband, middleband, lowerband

    def rsi(self, window_length=14):
        weighted_close = self.weighted_close_fxn()
        delta = weighted_close.diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = dUp.rolling(window=window_length).mean()
        RolDown = dDown.abs().rolling(window=window_length).mean()
        RS = RolUp / RolDown
        RSI = 100.0 - (100.0 / (1.0 + RS))
        return RSI

    def wwma(self, values, n):
        return values.ewm(alpha=1 / n, adjust=False).mean()

    def atr(self, n=7):
        data = pd.DataFrame()
        data["tr0"] = abs(self.high - self.low)
        data["tr1"] = abs(self.high - self.close.shift())
        data["tr2"] = abs(self.low - self.close.shift())
        tr = data[["tr0", "tr1", "tr2"]].max(axis=1)
        atr = self.wwma(tr, n)
        return atr

    def sma(self, period):
        weighted_close = self.weighted_close_fxn()
        smas = weighted_close.ewm(span=period, adjust=False).mean()
        return smas

    def ema(self, period):
        weighted_close = self.weighted_close_fxn()
        emas = weighted_close.ewm(span=period, adjust=False).mean()
        return emas

    def vwap(self, df):
        df["weighted_close"] = self.weighted_close_fxn()
        q = df.volume.values
        p = df.weighted_close.values
        return df.assign(vwap=(p * q).cumsum() / q.cumsum())

    def return_vwap(self):
        df = self.dataset.copy()
        df = df.groupby(df.index.date, group_keys=False).apply(self.vwap)
        return df["vwap"]

    def macd(self):
        weighted_close = self.weighted_close_fxn()
        sma1 = weighted_close.ewm(span=12, adjust=False).mean()
        sma2 = weighted_close.ewm(span=26, adjust=False).mean()
        dcam = sma1 - sma2
        signalmacd = dcam.ewm(span=9, adjust=False).mean()
        macdhist = dcam - signalmacd
        return dcam, signalmacd, macdhist

    def psar(self, iaf=0.0011, maxaf=0.2):
        length = len(self.dataset)
        high = self.high
        low = self.low
        close = self.close.copy()
        psar = close[0 : len(close)]
        psarbull = [None] * length
        psarbear = [None] * length
        bull = True
        af = iaf
        ep = low[0]
        hp = high[0]
        lp = low[0]
        for i in range(2, length):

            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            reverse = False
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + iaf, maxaf)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + iaf, maxaf)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]
        return psar

    def mom(self, period=24):
        weighted_close = self.weighted_close_fxn()
        x = weighted_close - weighted_close.shift(24)
        return x

    def stochastics(self, k=14, d=3):
        """
        Fast stochastic calculation
        %K = (Current Close - Lowest Low)/
        (Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

        Slow stochastic calculation
        %K = %D of fast stochastic
        %D = 3-day SMA of %K

        When %K crosses above %D, buy signal
        When the %K crosses below %D, sell signal
        """
        low = "low"
        high = "high"
        weighted_close = self.weighted_close_fxn()
        df = self.dataset.copy()
        df["weighted_close"] = weighted_close
        close = "weighted_close"

        # Set minimum low and maximum high of the k stoch
        low_min = df[low].rolling(window=k).min()
        high_max = df[high].rolling(window=k).max()

        # Fast Stochastic
        k_fast = 100 * (df[close] - low_min) / (high_max - low_min)
        d_fast = k_fast.rolling(window=d).mean()

        # Slow Stochastic
        k_slow = d_fast
        d_slow = k_slow.rolling(window=d).mean()

        return k_fast, d_fast, k_slow, d_slow

    def ichimoku_cloud(
        self,
        kijun_lag=48,
        tenkan_lag=9,
        chikou_lag=26,
        senkou_a_projection=26,
        senkou_b_lag=52,
    ):
        periodk_high = self.high.rolling(window=kijun_lag).max()
        periodk_low = self.high.rolling(window=kijun_lag).min()
        kijun_sen = (periodk_high + periodk_low) / 2
        periodt_high = self.high.rolling(window=tenkan_lag).max()
        periodt_low = self.high.rolling(window=tenkan_lag).min()
        tenkan_sen = (periodt_high + periodt_low) / 2
        chikou_span = self.close.shift(-chikou_lag)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(senkou_a_projection)
        period52_high = self.high.rolling(window=senkou_b_lag).max()
        period52_low = self.low.rolling(window=senkou_b_lag).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(chikou_lag)
        return kijun_sen, tenkan_sen, chikou_span, senkou_span_a, senkou_span_b


# Data Download and Setup
def request_download(url, headers=None):
    """Request to get the url passed and returns the response."""
    active = True
    count = 0
    wait_time = 10
    while active:
        try:
            if headers:
                res = requests.get(url, headers=headers)
            else:
                res = requests.get(url)
            res.raise_for_status()
        except requests.exceptions.RequestException as err:
            print(f"Couldn't Get Request: {err}", file=sys.stderr)
            if res:
                print(res)
            count += 1
            if count == 15:
                print(
                    "Process Terminated, Check Internet Connectivity and Try Again!!!",
                    file=sys.stderr,
                )
                return {"status_code": -1}
            for i in reversed(range(wait_time)):
                print(f"Retrying in {i}")
                time.sleep(1)
            wait_time += 3
        else:
            active = False
    for i in range(4):
        time.sleep(1)
    return res


def get_klines(symbol, interval, limit=1000):
    """Fetches kline data from the api used"""
    kline_data = (
        base_api + f"/api/v1/klines?symbol={symbol}&limit={limit}&interval={interval}"
    )
    res = request_download(kline_data)
    print(symbol, " downloaded, now saving...")
    print(res.status_code)
    if res.status_code != 200:
        print("An Error Occured")
        sys.exit(-1)  # change sys exit to -> return { "data": "FAILED GET")
    dataset = pd.DataFrame(
        res.json(),
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "quoteVolume",
            "num_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    dataset.drop(["ignore"], axis=1, inplace=True)
    dataset["openTime"] = pd.to_datetime(
        dataset["openTime"] * 1000000, utc="True"
    ).dt.tz_convert("Africa/Lagos")
    dataset["closeTime"] = pd.to_datetime(
        dataset["closeTime"] * 1000000, utc="True"
    ).dt.tz_convert("Africa/Lagos")
    return dataset


def download_symbols(symbols, interval="1h"):
    """Fetches and saves kline data gotten for symbols passed"""
    for symbol in symbols:
        symbol, interval = get_symbol_interval(symbol)
        dataset = get_klines(symbol, interval)
        symbol = symbol + interval
        all_data[symbol] = dataset
        time.sleep(2)


def set_up_data(symbol, vwap=1, win=24):
    """Sets up data table to be used for predictions with default values"""
    global all_data
    dataset = all_data[symbol].copy()
    print("setting up", symbol)
    # dataset.drop(dataset.tail(1).index,inplace=True)
    dataset["openTime"] = pd.to_datetime(dataset["openTime"])
    dataset["closeTime"] = pd.to_datetime(dataset["closeTime"])
    dataset["closecheck"] = pd.to_numeric(dataset["close"], errors="coerce")
    dataset["close"] = pd.to_numeric(dataset["open"], errors="coerce")
    dataset["high"] = pd.to_numeric(dataset["high"], errors="coerce")
    dataset["low"] = pd.to_numeric(dataset["low"], errors="coerce")
    dataset["open"] = pd.to_numeric(dataset["close"], errors="coerce")
    dataset["Close"] = pd.to_numeric(dataset["open"], errors="coerce")
    dataset["High"] = pd.to_numeric(dataset["high"], errors="coerce")
    dataset["Low"] = pd.to_numeric(dataset["low"], errors="coerce")
    dataset["Open"] = pd.to_numeric(dataset["close"], errors="coerce")
    dataset["Volume"] = pd.to_numeric(dataset["volume"], errors="coerce")
    dataset["volume"] = pd.to_numeric(dataset["volume"], errors="coerce")
    if vwap:
        dataset["close"] = ta.volume.volume_weighted_average_price(
            high=dataset["High"],
            low=dataset["Low"],
            close=dataset["Close"],
            volume=dataset["Volume"],
            window=win,
        )
    dataset.set_index("openTime", inplace=True)
    return dataset


def set_up_full_data(symbol, vwap=None, win=24):
    """Sets up data table to be used with both default values and values from indicators"""
    # To replace hard coded values with values from optional arguments: future work
    data = set_up_data(symbol, vwap=vwap, win=win)
    my_indicators = Fin_Indicator(data)
    weighted_close = my_indicators.weighted_close_fxn()
    data["weighted_close"] = weighted_close
    data["vwap"] = my_indicators.return_vwap()
    data["SMA1"] = my_indicators.sma(period=12)
    data["SMA2"] = my_indicators.sma(period=24)
    upperband, middleband, lowerband = my_indicators.bbands(typ="weighted_close")
    data["BBupperband"] = upperband
    data["BBlowerband"] = lowerband
    data["BBmiddleband"] = middleband
    data["EMASpan1"] = my_indicators.ema(period=12)
    data["EMASpan2"] = my_indicators.ema(period=24)
    data["psar"] = my_indicators.psar()
    (
        data["macd"],
        data["macdsignal"],
        data["macdhist"],
    ) = my_indicators.macd()
    data["rsi"] = my_indicators.rsi(window_length=24)
    data["atr"] = my_indicators.atr(n=7)
    data["mom"] = my_indicators.mom(period=24)
    (
        data["stochf_fastk"],
        data["stochf_fastd"],
        data["stoch_slowk"],
        data["stoch_slowd"],
    ) = my_indicators.stochastics()
    return data


