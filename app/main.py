from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlite3 import Connection as SQLite3Connection
from datetime import datetime
from sqlalchemy import event, JSON, ARRAY
from sqlalchemy.engine import Engine
from flask_sqlalchemy import SQLAlchemy
import time

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sqlitedb.file"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = 0

# configure sqlite3 to enforce foreign key constraints
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


# create database instance
db = SQLAlchemy(app)
now = datetime.now()

# models
class Prediction(db.Model):
    __tablename__ = "Prediction"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50))
    date_created = db.Column(db.Date)
    data = db.Column(JSON)

class Backtest(db.Model):
    __tablename__ = "Backtest"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50))
    symbol_strategy = db.Column(db.String(50))
    strategy = db.Column(db.String(50))
    date_created = db.Column(db.Date)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    percentage_exposure_time = db.Column(db.Float)
    percentage_returns = db.Column(db.Float)
    percentage_buy_and_hold_return = db.Column(db.Float)
    maximum_drawdown = db.Column(db.Float)
    maximum_drawdown_duration = db.Column(db.String(50))
    num_trades = db.Column(db.Integer)
    win_rate = db.Column(db.Float)
    trade_percentage_expectancy = db.Column(db.Float)
    profit_factor = db.Column(db.Float)

# Create the table
db.drop_all()
try:
  db.create_all()
except:
  print(Prediction.query.all(), "Prediction List")

print("Main Up and Running...")

@app.route("/predictions", methods=["GET"])
def get_predictions():
    print("Getting Predictions")
    predictions = Prediction.query.all()
    return_list = []
    print(predictions)
    for prediction in predictions:
        return_list.append(
            {"id": prediction.id, 
            "symbol": prediction.symbol, 
            "data": prediction.data,
            "date_created": prediction.date_created
            }
        )
    return jsonify(return_list), 200

@app.route("/backtests", methods=["GET"])
def get_backtests():
    backtests = Backtest.query.all()
    return_list = []
    for backtest in backtests:
        return_list.append(
            {
                "id": backtest.id,
                "date_created": backtest.date_created,
                "symbol": backtest.symbol,
                "symbol_strategy": backtest.symbol_strategy,
                "strategy": backtest.strategy,
                "start_date": backtest.start_date,
                "end_date": backtest.end_date,
                "percentage_returns": backtest.percentage_returns,
                "percentage_buy_and_hold_return": backtest.percentage_buy_and_hold_return,
                "maximum_drawdown": backtest.maximum_drawdown,
                "maximum_drawdown_duration": backtest.maximum_drawdown_duration,
                "num_trades": backtest.num_trades,
                "win_rate": backtest.win_rate,
                "trade_percentage_expectancy": backtest.trade_percentage_expectancy,
                "profit_factor": backtest.profit_factor,
                "percentage_exposure_time": backtest.percentage_exposure_time,
            }
        )
    return jsonify(return_list), 200


@app.route("/predictions/<symbol>", methods=["GET"])
def get_prediction(symbol):
    predictions = Prediction.query.all()
    return_list = []
    for prediction in predictions:
        if prediction.symbol == symbol:
            return_list.append(
            {"id": prediction.id, 
            "symbol": prediction.symbol, 
            "data": prediction.data,
            "date_created": prediction.date_created
            }
            )
    prediction = return_list[0] if len(return_list) else None
    return jsonify(prediction), 200


@app.route("/backtests/<symbol_strategy>", methods=["GET"])
def get_backtest(symbol_strategy):
    backtests = Backtest.query.all()
    return_list = []
    for backtest in backtests:
        if backtest.symbol_strategy == symbol_strategy:
            return_list.append(
                {
                    "id": backtest.id,
                    "date_created": backtest.date_created,
                    "symbol": backtest.symbol,
                    "symbol_strategy": backtest.symbol_strategy,
                    "strategy": backtest.strategy,
                    "start_date": backtest.start_date,
                    "end_date": backtest.end_date,
                    "percentage_returns": backtest.percentage_returns,
                    "percentage_buy_and_hold_return": backtest.percentage_buy_and_hold_return,
                    "maximum_drawdown": backtest.maximum_drawdown,
                    "maximum_drawdown_duration": backtest.maximum_drawdown_duration,
                    "num_trades": backtest.num_trades,
                    "win_rate": backtest.win_rate,
                    "trade_percentage_expectancy": backtest.trade_percentage_expectancy,
                    "profit_factor": backtest.profit_factor,
                    "percentage_exposure_time": backtest.percentage_exposure_time,
                }
            )

    backtest = return_list[0] if len(return_list) else None
    return jsonify(backtest), 200
