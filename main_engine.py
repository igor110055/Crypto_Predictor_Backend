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
exchange_data = ""

# All : future work should take speed of executions into account, future work, in itself, can also be to speed up execution time

# General Functions
def get_symbol_interval(symbol):
    """Given a 'symbol' eg -> 'BTCUSDT1h'
    it returns -> 'BTCUSDT', '1h'
    """
    if "30m" in symbol:
        return symbol.replace("30m", ""), "30m"
    elif "1h" in symbol:
        return symbol.replace("1h", ""), "1h"
    elif "12h" in symbol:
        return symbol.replace("12h", ""), "12h"
    elif "2h" in symbol:
        return symbol.replace("2h", ""), "2h"
    elif "4h" in symbol:
        return symbol.replace("4h", ""), "4h"
    elif "6h" in symbol:
        return symbol.replace("6h", ""), "6h"
    elif "1D" in symbol:
        return symbol.replace("1D", ""), "1D"
    else:
        return symbol, "1h"


# Main Engine for Indicators used. To be replaced with ta library: future work
class Fin_Indicator:
    """Class definition for various indicators used in this project."""

    def __init__(self, main_dataset):
        self.dataset = main_dataset.copy()
        self.open_ = self.dataset["open"]
        self.high = self.dataset["high"]
        self.low = self.dataset["low"]
        self.close = self.dataset["close"]
        self.volume = self.dataset["volume"]

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

    def ema(self, period, weights=0):
        if weights:
            close = self.weighted_close_fxn()
        else:
            close = self.close
        emas = close.ewm(span=period, adjust=False).mean()
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

    def macd(self, int1, int2, macdint, weights=0):
        """int1 - interval/period for the first sma
        int2 - interval for the second sma
        macdint - interval for the macd
        weights - to use weighted close or not.
        """
        # intervals to be replaced with a function that determines the best interval to use, as well as what attribute to use between com, halflife and span(span is used for now as is below): future work
        if weights:
            close = self.weighted_close_fxn()
        else:
            close = self.close
        sma1 = close.ewm(span=int1, adjust=False).mean()
        sma2 = close.ewm(span=int2, adjust=False).mean()
        dcam = sma1 - sma2
        signalmacd = dcam.ewm(span=macdint, adjust=False).mean()
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
            count += 1
            if count == 1:
                print(
                    "Process Terminated, Check Internet Connectivity and Try Again!!!",
                    file=sys.stderr,
                )
                return
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
    if (not res) or (res.status_code != 200):
        print("An Error Occured")
        sys.exit(-1)  # change sys exit to -> return { "data": "FAILED GET")
    print(symbol, " downloaded, now saving...")
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


def multi_thread_download(my_symbols):
    """Multi threads the download of multiple symbols"""
    downloadThreads = []
    adder = int(len(my_symbols) / 6)
    v = 0
    for i in range(0, len(my_symbols), adder):
        symbols_list_truncated = my_symbols[v : i + adder]
        downloadThread = threading.Thread(
            target=download_symbols, args=([symbols_list_truncated])
        )
        downloadThreads.append(downloadThread)
        downloadThread.start()
        v += adder
    for downloadThread in downloadThreads:
        downloadThread.join()
    print("Done.")


def set_up_data(symbol, vwap=1, win=24):
    # Explain or simplify code later: future work
    """Sets up data table to be used for predictions with default values"""
    # uncomment later
    # global all_data
    # dataset = all_data[symbol].copy()

    # comment later
    path = "/home/ihechi/Documents/Datasheets_and_Datasets/crypto/Bot/"
    pkl_file = open(path + symbol + ".pkl", "rb")
    dataset = pickle.load(pkl_file)
    pkl_file.close()

    print("setting up", symbol)
    dataset["openTime"] = pd.to_datetime(dataset["openTime"])
    dataset["closeTime"] = pd.to_datetime(dataset["closeTime"])
    dataset["closecheck"] = pd.to_numeric(dataset["close"], errors="coerce")
    dataset["close"] = pd.to_numeric(dataset["open"], errors="coerce")
    dataset["high"] = pd.to_numeric(dataset["high"], errors="coerce")
    dataset["low"] = pd.to_numeric(dataset["low"], errors="coerce")
    dataset["open"] = pd.to_numeric(dataset["close"], errors="coerce")
    dataset["opencheck"] = pd.to_numeric(dataset["open"], errors="coerce")
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

    # calculates and sets up heiken data
    datasets = dataset[["opencheck", "High", "Low", "closecheck"]]
    datasets.columns = ["Open", "High", "Low", "Close"]
    datasets_ha = datasets.copy()
    for i in range(datasets_ha.shape[0]):
        if i > 0:
            datasets_ha.loc[datasets_ha.index[i], "Open"] = (
                datasets["Open"][i - 1] + datasets["Close"][i - 1]
            ) / 2
            datasets_ha.loc[datasets_ha.index[i], "Close"] = (
                datasets["Open"][i]
                + datasets["Close"][i]
                + datasets["Low"][i]
                + datasets["High"][i]
            ) / 4
    datasets_ha = datasets_ha.iloc[1:, :]
    dataset[["heiken_open", "heiken_high", "heiken_low", "heiken_close"]] = datasets_ha[
        ["Open", "High", "Low", "Close"]
    ]

    return dataset


def set_up_full_data(
    symbol, fai=0.0011, afmax=0.2, vwap=None, win=14, typ="weighted_close"
):
    """Sets up data table to be used with both default values and some values from indicators"""
    # To replace hard coded values with values from optional arguments: future work
    dataset = set_up_data(symbol, vwap=vwap, win=win)
    my_indicators = Fin_Indicator(dataset)
    weighted_close = my_indicators.weighted_close_fxn()
    dataset["weighted_close"] = weighted_close
    dataset["vwap"] = my_indicators.return_vwap()
    dataset["SMA1"] = my_indicators.sma(period=12)
    dataset["SMA2"] = my_indicators.sma(period=24)
    upperband, middleband, lowerband = my_indicators.bbands(typ=typ)
    dataset["BBupperband"] = upperband
    dataset["BBlowerband"] = lowerband
    dataset["BBmiddleband"] = middleband
    dataset["EMASpan1"] = my_indicators.ema(period=12)
    dataset["EMASpan2"] = my_indicators.ema(period=24)
    dataset["psar"] = my_indicators.psar(iaf=fai, maxaf=afmax)
    (
        dataset["macd"],
        dataset["macdsignal"],
        dataset["macdhist"],
    ) = my_indicators.macd(int1=6, int2=24, macdint=9)
    dataset["rsi"] = my_indicators.rsi(window_length=24)
    dataset["atr"] = my_indicators.atr(n=7)
    dataset["mom"] = my_indicators.mom(period=24)
    (
        dataset["stochf_fastk"],
        dataset["stochf_fastd"],
        dataset["stoch_slowk"],
        dataset["stoch_slowd"],
    ) = my_indicators.stochastics()
    return dataset


# Strategy building functions
# To Explain what the codes do or ensure they are self explanatory: future work


def return_best_rsi(symbol):
    """Returns the best rsi values for a symbol"""
    dataset = set_up_data(symbol)
    my_indicators = Fin_Indicator(dataset)
    data = dataset.copy()
    sma1 = range(25, 46, 5)
    sma2 = range(55, 86, 5)
    results_list = []
    v = pd.DataFrame()
    for SMA1, SMA2 in product(sma1, sma2):
        #         data = dataset.copy()
        data = (pd.DataFrame(data["close"])).dropna()
        data.dropna(inplace=True)
        data["Returns"] = np.log(data["close"] / data["close"].shift(1))
        #         v["Returns"] = data["Returns"]
        weighted_close = my_indicators.weighted_close_fxn()
        data["rsi"] = my_indicators.rsi(window_length=24)
        data["Position"] = np.where((data["rsi"] < SMA1), 1, np.nan)
        data["Position"] = np.where((data["rsi"] > SMA2), -1, data["Position"])
        data["Position"].ffill(inplace=True)
        data["Strategy"] = data["Position"].shift(1) * data["Returns"]
        perf = np.exp(data[["Returns", "Strategy"]].sum())
        results_list.append(
            {
                "SMA1": SMA1,
                "SMA2": SMA2,
                "Market": perf["Returns"],
                "Strategy": perf["Strategy"],
                "OUT": perf["Strategy"] - perf["Returns"],
            }
        )
    results = pd.DataFrame.from_records(results_list)
    return results.sort_values("OUT", ascending=False).iloc[0]


def trail_stop(symbol, multiplier, vwap=None, win=14):
    """Returns trailing stop values with position prediction"""
    # To write explanations for what this function does in detail: future work
    dataset = set_up_full_data(symbol, vwap=vwap, win=win)
    my_indicators = Fin_Indicator(dataset)
    vc = pd.DataFrame()
    vc["atr"] = dataset["atr"]
    vc["close"] = dataset["close"]
    ac = []
    vc.dropna(inplace=True)
    vc["rsi"] = my_indicators.rsi(window_length=24)
    smas = return_best_rsi(symbol)
    SMA1 = int(smas["SMA1"])
    SMA2 = int(smas["SMA2"])
    vc["close"] = np.where(
        vc["rsi"] > SMA2, vc["close"] + (vc["close"] * 0.0098), vc["close"]
    )
    vc["close"] = np.where(
        vc["rsi"] < SMA1, vc["close"] - (vc["close"] * 0.0098), vc["close"]
    )
    ac.append(vc["close"][0] - (vc["atr"][0] * multiplier))
    for index, rows in vc.iterrows():
        if ac[-1] < rows["close"]:
            patr = rows["close"] - (rows["atr"] * multiplier)
            if patr > ac[-1]:
                ac.append(patr)
            else:
                ac.append(ac[-1])
        else:
            patr = rows["close"] + (rows["atr"] * 3.1)
            if patr < ac[-1]:
                ac.append(patr)
            else:
                ac.append(ac[-1])
    del ac[0]
    kd = pd.Series(ac, index=vc.index)
    trail_stop_df = pd.DataFrame()
    trail_stop_df["close"] = vc["close"]
    trail_stop_df["atr2"] = kd
    trail_stop_df["exit_signal"] = trail_stop_df["close"] < trail_stop_df["atr2"]
    trail_stop_df["Position"] = np.where(trail_stop_df["exit_signal"], -1, np.nan)
    return trail_stop_df["Position"], trail_stop_df["atr2"]


# Main Predictor


def calc_all(symbol, stop, kijun, fai=0.0011, vwap=None, win=14):
    """Returns a dataset computed with predicted positions and the name of the best strategy column

    stop: trail stop multiplier value
    win: window period for most indicators calculation
    fai: is iaf - increment acceleration factor
    """
    # set up dataset
    dataset = set_up_full_data(symbol, fai=fai, vwap=vwap, win=win)

    # initialize the Fin_Indicator class
    my_indicators = Fin_Indicator(dataset)

    # create a copy of the dataset for data manipulation
    dataset_copy = dataset.copy()

    # set up a new dataframe to hold values for the strategies and other useful data
    clean_data = pd.DataFrame(
        dataset[
            [
                "close",
                "weighted_close",
                "closeTime",
                "low",
                "high",
                "closecheck",
                "opencheck",
                "vwap",
                "open",
                "macd",
                "macdsignal",
                "macdhist",
                "psar",
                "BBmiddleband",
            ]
        ].copy()
    )
    (
        clean_data["kijun_sen"],
        clean_data["tenkan_sen"],
        clean_data["chikou_span"],
        clean_data["senkou_span_a"],
        clean_data["senkou_span_b"],
    ) = my_indicators.ichimoku_cloud(kijun_lag=kijun)

    # MACD Strategy
    #  check EMA Strategy to set up macd and macdsignals using preferred values
    dataset_copy = dataset.copy()

    dataset_copy["Position"] = np.where(
        dataset_copy["macd"] > dataset_copy["macdsignal"], 1, -1
    )
    dataset_copy["Position"].bfill(inplace=True)
    dataset_copy["Position"].ffill(inplace=True)
    dataset_copy["Position"] = pd.Series(
        np.nan_to_num(dataset_copy["Position"]),
        index=dataset_copy.index,
        name="Position",
    )

    clean_data["PositionMACD"] = dataset_copy["Position"]

    # EMA Strategy
    dataset_copy = dataset.copy()
    SMA1 = 6  # interval 1, both intervals to be replaced with a function that determines the best interval to use, as well as what attributes to use between com, halflife and span(span is used for now as is below): future work
    SMA2 = 28  # interval 2

    dataset_copy["SMA1"] = my_indicators.ema(period=SMA1, weights=1)
    dataset_copy["SMA2"] = my_indicators.ema(period=SMA2, weights=1)

    dataset_copy["Position"] = np.where(
        dataset_copy["SMA1"] > dataset_copy["SMA2"], 1, -1
    )
    dataset_copy["Position"].bfill(inplace=True)
    dataset_copy["Position"].ffill(inplace=True)
    dataset_copy["Position"] = pd.Series(
        np.nan_to_num(dataset_copy["Position"]),
        index=dataset_copy.index,
        name="Position",
    )

    clean_data["PositionEMA"] = dataset_copy["Position"]

    # trail_stop Strategy and atr value
    clean_data["trail_stop"], clean_data["atr2"] = trail_stop(symbol, stop)

    # Strategy2 Strategy
    clean_data["Strategy2"] = np.where((clean_data["PositionEMA"] == 1), 1, np.nan)
    clean_data["Strategy2"] = np.where(
        (clean_data["trail_stop"] == -1) & (clean_data["PositionMACD"] == -1),
        -1,
        clean_data["Strategy2"],
    )

    # PostionPR Strategy
    clean_data["PositionPR"] = np.where(clean_data["psar"] > clean_data["close"], -1, 1)

    # prX Strategy
    clean_data["prX"] = np.where(
        (clean_data["Strategy2"] == 1) & (clean_data["PositionPR"] == 1), 1, -1
    )

    # prXX Strategy
    clean_data["prXX"] = np.where(
        (clean_data["prX"] == 1)
        & (clean_data["low"] < clean_data["BBmiddleband"])
        & (clean_data["high"] > clean_data["BBmiddleband"])
        & (clean_data["closecheck"] > clean_data["opencheck"]),
        1,
        np.nan,
    )
    clean_data["prXX"] = np.where(
        (clean_data["prX"] == -1)
        & (clean_data["low"] < clean_data["BBmiddleband"])
        & (clean_data["high"] > clean_data["BBmiddleband"])
        & (clean_data["closecheck"] < clean_data["opencheck"]),
        -1,
        clean_data["prXX"],
    )

    # cmf_ii Strategy
    clean_data["trend_aroon_up"] = ta.trend.aroon_up(
        close=clean_data["closecheck"], window=12
    )
    clean_data["trend_aroon_down"] = ta.trend.aroon_down(
        close=clean_data["closecheck"], window=12
    )

    clean_data["cmf_ii"] = np.where(
        (clean_data["trend_aroon_up"] > clean_data["trend_aroon_down"])
        & (clean_data["trend_aroon_up"] > 95),
        1,
        np.nan,
    )
    clean_data["cmf_ii"] = np.where(
        (clean_data["trend_aroon_up"] < clean_data["trend_aroon_down"])
        & (clean_data["trend_aroon_down"] > 95),
        -1,
        clean_data["cmf_ii"],
    )

    clean_data["cmf_ii"].ffill(inplace=True)

    clean_data["prXXcmf"] = np.where(
        (clean_data["prXX"] == 1) & (clean_data["cmf_ii"] == 1), 1, -1
    )

    # forward and backward fills nan values
    clean_data.ffill(inplace=True)
    clean_data.bfill(inplace=True)

    best = "prXX"  # column name for best strategy at the moment.

    return clean_data, best


my_symbols = [
    "BTCUSDT15m",
    "ETHUSDT1h",
    "BNBUSDT1h",
    "LUNABUSD1h",
    "SOLUSDT1h",
    "CHZUSDT1h",
    "SANDUSDT1h",
]
print(all_data)
x = calc_all(my_symbols[0], 2.5, 48, vwap=1, win=24)
print(x[0][["weighted_close", "Strategy2", "prXX", "PositionPR", "cmf_ii"]][-50:])
print(all_data)
