# Crypto_Predictor_Backend

- Api Services

Built this for a crypto predictor website I intend to build soon. It is still a work in progress, as such, it is currently lacking in documentation. Right now, most features are purely experimental as I don't have any experience building a flask API and this is my first. Please check back later. I intend to finish this off once I complete the front end part.

Issues to work on:

- Check the Issues tab

Current Progess:

The current state of the API can be seen through the following routes.

- [View Predictions](https://get-crypto-predicts.herokuapp.com/predictions)
- [View Backtests](https://get-crypto-predicts.herokuapp.com/backtests)
- [View Predictions for a particular symbol](https://get-crypto-predicts.herokuapp.com/predictions/BTCUSDT1h)
  - Symbol name here is in form of 'symbol' + 'interval'
- [View Backtest Results for a particular strategy](https://get-crypto-predicts.herokuapp.com/backtests/BTCUSDT1h_PositionMACD)
  - Strategy name here is in form of 'symbol' + 'interval' + '\_' + 'strategy'

Symbols currently supported:

["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "MATICUSDT", "THETAUSDT", "CHZUSDT"]

Intervals Currently supported:

["1h", "2h", "4h", "8h", "12h", "1d"]

Current Strategies(Names):

["PositionMACD", "PositionEMA", "trail_stop", "Strategy2", "PositionPR", "prX", "prXX", "cmf", "prXXcmf"]
