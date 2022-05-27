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


symbols_list = ["ENJUSDT", "BTCUSDT", "EOSUSDT", "ONTUSDT", "XRPUSDT", "AXSUSDT", "CHZUSDT","ETHUSDT","BNBUSDT", "MATICUSDT"]
# intervals = ["1h", "15m", "30m", "1h", "1d", "5m", "4h"]
intervals = ["1h", "30m", "2h", "4h", "1d", "12h"]
strategies = ["PositionMACD", "PositionEMA", "trail_stop", "Strategy2", "PositionPR", "prX", "prXX", "cmf", "prXXcmf"]

start = datetime.now()
results, calculated_data = get_results(symbols_list[:2], intervals[:1], strategies[:1])
print(results)
print(type(results))
for symbol, dataset in results.items():
  now = datetime.now()
  print(symbol, ": Working on!")
  new_prediction = application.Prediction(
    symbol=symbol, created_at=now #dataset.astype(str)[:150] # .to_dict('dict')
  )
  db.session.add(new_prediction)
  db.session.commit()

print("Done!")
exit()