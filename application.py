from flask import Flask, request, jsonify
from sqlite3 import Connection as SQLite3Connection
from datetime import datetime
from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask_sqlalchemy import SQLAlchemy
import time

app = Flask(__name__)

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
  created_at = db.Column(db.Date)
  data = db.Column(db.String(200))

class Backtest(db.Model):
  __tablename__ = "Backtest"
  id = db.Column(db.Integer, primary_key=True)
  symbol_strategy = db.Column(db.String(50))
  created_at = db.Column(db.Date())
  data = db.Column(db.String(200))


# Create the table
db.drop_all()
db.create_all()

# models
# class Backtest(db.Model):
#   __tablename__ = "Backtest"
#   id = db.Column(db.Integer, primary_key=True)
#   name = db.Column(db.String(50))
#   email = db.Column(db.String(50))
#   address = db.Column(db.String(200))
#   phone = db.Column(db.String(50))
#   posts = db.relationship("BlogPost", cascade=("all, delete"))

@app.route('/predictions', methods=["GET"])
def get_predictions():
  predictions = Prediction.query.all()
  return_list = []
  for prediction in predictions: 
    return_list.append(
      {
        "id": prediction.id,
        "symbol": prediction.symbol,
        "data": prediction.data
      }
    )
  return jsonify(return_list), 200

@app.route('/backtests', methods=["GET"])
def get_backtests():
  backtests = Backtest.query.all()
  return_list = []
  for backtest in backtests: 
    return_list.append(
      {
        "id": backtest.id,
        "symbol": backtest.symbol,
        "data": backtest.data
      }
    )
  return jsonify(return_list), 200

@app.route('/predictions/<symbol>', methods=["GET"])
def get_prediction(symbol):
  predictions = Prediction.query.all()
  return_list = []
  for prediction in predictions: 
    if prediction.symbol == symbol:
      return_list.append(
        {
          "id": prediction.id,
          "symbol": prediction.symbol,
          "data": prediction.data
        }
    )
  
  
  prediction = return_list[0] if len(return_list) else None
  print(prediction)
  return jsonify(prediction), 200

if __name__ == "__main__":
  app.run(debug=True)