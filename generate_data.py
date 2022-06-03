from random import randrange
from sqlite3 import Connection as SQLite3Connection
from datetime import datetime, timedelta
from sqlalchemy import event, create_engine
from sqlalchemy.engine import Engine
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import time
import random


from app.main_engine import get_results
import app.main as application

dataset_num = 300  # Number of Prediction Data points to send to database. Currently, the number of data points is 1000: set by limit in main_engine.get_klines function.

def generate_data():
    print("Generating New data")
    symbols_list = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "MATICUSDT",
        "THETAUSDT",
        "CHZUSDT",
    ]
    # intervals = ["1h", "15m", "30m", "1h", "1d", "5m", "4h"]
    intervals = ["1h", "2h", "4h", "8h", "12h", "1d"]
    strategies = [
        "MACDStrategy",
        "EMAStrategy",
        "TrailStopATRStrategy",
        "PSARStrategy",
        "TEMStrategy",
        "PTStrategy",
        "XStrategy",
        "AroonStrategy",
        "XAroonStrategy",
    ]

    results, calculated_data = get_results(
        symbols_list[:3], intervals[:1], strategies[:1]
    )  # check

    # Drop existing data and create new tables so the database only contains fresh values
    application.db.drop_all()
    application.db.create_all()

    for symbol, dataset in calculated_data.items():
        now = datetime.now()
        new_prediction = application.Prediction(
            date_created=now,
            symbol=symbol,
            data=dataset[len(dataset) - dataset_num :].to_dict(
                "list"
            ),  # send only 300 data points to database. Most Recent
        )
        application.db.session.add(new_prediction)
        application.db.session.commit()

    for symbol_strategy, dataset in results.items():
        now = datetime.now()
        new_backtest = application.Backtest(
            date_created=now,
            symbol_strategy=symbol_strategy,
            symbol=symbol_strategy.split("_")[0],
            strategy=symbol_strategy.split("_")[1],
            start_date=dataset["start_date"],
            end_date=dataset["end_date"],
            percentage_exposure_time=dataset["percentage_exposure_time"],
            percentage_returns=dataset["percentage_returns"],
            percentage_buy_and_hold_return=dataset["percentage_buy_&_hold_return"],
            maximum_drawdown=dataset["maximum_drawdown"],
            maximum_drawdown_duration=str(dataset["maximum_drawdown_duration"]),
            num_trades=dataset["num_trades"],
            win_rate=dataset["win_rate"],
            trade_percentage_expectancy=dataset["trade_percentage_expectancy"],
            profit_factor=dataset["profit_factor"],
        )
        application.db.session.add(new_backtest)
        application.db.session.commit()

    print("Done!")


def setup_data_generate():
    print("Generating...")
    # To replace this with apscheduler
    active = True
    while active:
        time.sleep(1)
        current_time = datetime.now()
        current_minute = current_time.minute
        if current_minute < 5:
            active = False
    # The block of code above helps me ensure the code starts running atleast some minutes close
    # to the nearest hr. pending when replaced with apscheduler. It waits until the time is within the first 0 - 4 minutes of the closest hr.

    active = True
    while active:
        start = datetime.now()
        for i in range(
            240
        ):  # wait additional 4 minutes cause sometimes, updates from binance take time.
            print("Waiting for 4 minutes.")
            time.sleep(1)
        generate_data()
        end = datetime.now()
        time_taken = (end - start).seconds
        wait_time = timedelta(
            hours=1
        ).seconds  # code should run every 1 hr(lowest time interval) to generate newer data
        handle_miniseconds = random.choice([2, -2])
        actual_wait_time = (
            wait_time - time_taken
        ) + handle_miniseconds  # minus or add 2 seconds to handle miniseconds. this way, the time difference always remain close to the actual wait time needed
        print(
            "Next Update: ",
            datetime.now() + timedelta(seconds=actual_wait_time),
            "seconds, took: ",
            time_taken,
        )
        for i in range(actual_wait_time):
            time.sleep(1)


if __name__ == "__main__":
    print("Generating...")
    generate_data()
