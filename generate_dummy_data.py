from random import randrange
from sqlite3 import Connection as SQLite3Connection
from datetime import datetime
from main_engine import get_results
from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import application

# app
app = Flask(__name__)

# config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sqlitedb.file"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = 0

# configure sqlite3 to enforce foreign key contraints
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


db = SQLAlchemy(app)
now = datetime.now()

# faker = Faker()

import pandas as pd
symbols_list = ["ENJUSDT", "BTCUSDT", "EOSUSDT", "ONTUSDT", "XRPUSDT", "AXSUSDT", "CHZUSDT","ETHUSDT","BNBUSDT", "MATICUSDT"]
# intervals = ["1h", "15m", "30m", "1h", "1d", "5m", "4h"]
intervals = ["1h", "30m", "2h", "4h", "1d", "12h"]
strategies = ["PositionMACD", "PositionEMA", "trail_stop", "Strategy2", "PositionPR", "prX", "prXX", "cmf", "prXXcmf"]

results, calculated_data = get_results(symbols_list[:2], intervals[:1], strategies[:1])

for symbol_strategy, dataset in results.items():
  now = datetime.now()
  new_backtest = application.Backtest(
    symbol_strategy=symbol_strategy, 
    symbol=symbol_strategy.split("_")[0], 
    created_at=now,
    start_date = dataset['start_date'],
    end_date = dataset['end_date'],
    percentage_exposure_time = dataset["percentage_exposure_time"],

    percentage_returns = dataset['percentage_returns'],
    percentage_buy_and_hold_return = dataset['percentage_buy_&_hold_return'],

    maximum_drawdown = dataset['maximum_drawdown'],
    maximum_drawdown_duration = str(dataset['maximum_drawdown_duration']),
    num_trades = dataset['num_trades'],
    win_rate = dataset['win_rate'],
    trade_percentage_expectancy = dataset['trade_percentage_expectancy'],
    profit_factor = dataset['profit_factor'],
    #dataset.astype(str)[:150] # .to_dict('dict')
  )
  db.session.add(new_backtest)
  db.session.commit()

print("Done!")
exit()