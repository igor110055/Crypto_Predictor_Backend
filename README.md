# Crypto_Predictor_Backend

- Api Services

Built for [this crypto predictor website](https://github.com/The-Professor99/Crypto_Predictor_Frontend). It is still a work in progress, as such, it is currently lacking in documentation. Right now, most features are purely experimental as I don't have any experience building a flask API and this is my first. Please check back later. I intend to finish this off once I complete the front end part.

## Installation

This project can be installed by running the following commands in your preferred directory.

    $ git clone https://github.com/The-Professor99/Crypto_Predictor_Backend.git

## Requirements

<strong>This assumes you have <i>Python</i> installed on your system</strong>

[Setup a virtual environment and activate it](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Run the command below in the project's root folder to install the requirements.

    $ pip install -r requirements.txt

## How To Use

[Activate your virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Run the below command to start up the project

    $ python setup.py

- the server will run on port 5000: `http://localhost:5000/`. Check [Current Progress](#current-progess) for routes and socketIO connections that can be served.

Test data has already been generated, to generate new data, simply run

    $ python schedule_data_generate.py

- However, do note that this only generates data at intervals of 9:02am, 10:02am, 11:02am and so on...

## Issues to work on:

- Check the Issues tab
- Optimizing the algorithm [housed here](./app/prediction_engine/main_engine.py) for faster executions

## Current Progess:

The current state of the API can be seen through the following routes. [Please ensure that the server is running](#how-to-use) before visiting the following routes.

Routes

- [View Predictions](http://localhost:5000/predictions)
- [View Backtests](http://localhost:5000/backtests)
- [View Predictions for a particular symbol](http://localhost:5000/predictions/BTCUSDT1h)
  - Symbol name here is in form of 'symbol' + 'interval'
- [View Backtest Results for a particular strategy](http://localhost:5000/backtests/BTCUSDT1h_PositionMACD)
  - Strategy name here is in form of 'symbol' + 'interval' + '\_' + 'strategy'

Example:
To view an example of what the server returns without running it on a local dev environment, visit:

- [View Predictions for a particular symbol](https://get-crypto-predicts.herokuapp.com/predictions/BTCUSDT1h)

  - Symbol name here is in form of 'symbol' + 'interval'
  - Do note, however, that the herokuapp version currently deployed is outdated and may not have the same results as that of running a dev server.
  - The routes above can also be accessed through [this herokuapp link](https://get-crypto-predicts.herokuapp.com/)

- SocketIO connection is now supported. [Please go through this file](./app/main.py) for socket events currently supported pending when a proper documentation is made.

## Symbols currently supported:

["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "MATICUSDT", "THETAUSDT", "CHZUSDT"]

## Intervals Currently supported:

["1h", "2h", "4h", "8h", "12h", "1d"]

## Current Strategies(Names):

- When accessing the herokuapp
  ["PositionMACD", "PositionEMA", "trail_stop", "Strategy2", "PositionPR", "prX", "prXX", "cmf", "prXXcmf"]

- When running on a local dev environment
  ["MACDStrategy", "EMAStrategy", "TrailStopATRStrategy", "PSARStrategy", "TEMStrategy", "PTStrategy", "XStrategy", "AroonStrategy", "XAroonStrategy"]

## Disclaimer

The project is highly experimental and purely for learning/practise purposes.
You are entirely liable for any financial loss resulting from the use of this project for financial decisions.
