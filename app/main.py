from threading import Lock
from flask import (
    Flask,
    render_template,
    session,
    request,
    copy_current_request_context,
    jsonify,
)
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
from sqlite3 import Connection as SQLite3Connection
from datetime import datetime
from sqlalchemy import event, JSON, ARRAY
from sqlalchemy.engine import Engine
from flask_sqlalchemy import SQLAlchemy
import time
import app.helpers as helpers

from app.prediction_engine.main_engine import get_results

app = Flask(__name__)
CORS(app)


app.config["SECRET_KEY"] = "secret!"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sqlitedb.file"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = 0

socketio = SocketIO(app, cors_allowed_origins="*")
thread = None
thread_lock = Lock()

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
    date_created = db.Column(db.DateTime)
    data = db.Column(JSON)


class Backtest(db.Model):
    __tablename__ = "Backtest"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50))
    symbol_strategy = db.Column(db.String(50))
    strategy = db.Column(db.String(50))
    date_created = db.Column(db.DateTime)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    percentage_exposure_time = db.Column(db.Float)
    percentage_returns = db.Column(db.Float)
    percentage_buy_and_hold_return = db.Column(db.Float)
    maximum_drawdown = db.Column(db.Float)
    maximum_drawdown_duration = db.Column(db.String(50))
    num_trades = db.Column(db.Integer)
    win_rate = db.Column(db.Float)
    trade_percentage_expectancy = db.Column(db.Float)
    profit_factor = db.Column(db.Float)


def background_thread():
    count = 1
    print("*****" * 5)
    print("Counting", count)
    print("*****" * 5)
    while True:
        # Calculate time to nearest hr as the data should refresh every hr.
        minutes_to_nearest_hour = helpers.get_minutes_to_nearest_hr()

        # Socketio sleeps for as long as required
        # for i in range(minutes_to_nearest_hour * 6):
        socketio.sleep(10)

        print("Starting now: ", datetime.now())

        # Generate new data every hr
        # helpers.generate_data()

        # get the data generated from the database
        all_predictions = helpers.get_database_data("Prediction")
        all_backtests = helpers.get_database_data("Backtest")

        # Emit Data 
        socketio.emit("predictions", {"data": all_predictions, "count": count})
        socketio.emit("backtests", {"data": all_backtests, "count": count})
        count += 1


# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictions", methods=["GET"])
def get_predictions():
    return_list = helpers.get_database_data("Prediction")
    return jsonify(return_list), 200


@app.route("/backtests", methods=["GET"])
def get_backtests():
    return_list = helpers.get_database_data("Backtest")
    return jsonify(return_list), 200


@app.route("/predictions/<symbol>", methods=["GET"])
def get_prediction(symbol):
    return_list = helpers.get_database_data("Prediction", symbol)
    prediction = return_list[0] if len(return_list) else None
    return jsonify(prediction), 200


@app.route("/backtests/<symbol_strategy>", methods=["GET"])
def get_backtest(symbol_strategy):
    return_list = helpers.get_database_data("Backtest", symbol_strategy)
    backtest = return_list[0] if len(return_list) else None
    return jsonify(backtest), 200


# Socket IO Events


@socketio.event
def connect():
    print("Client Connected", request)
    typ = request.args.get("type")
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

    emit("connect_success", {"data": "Connected"})


@socketio.on("disconnect")
def test_disconnect():
    print("Client disconnected", request.sid)
    disconnect()
