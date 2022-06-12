from datetime import datetime
from typing import List, Union
import app.main as main_app
from app.prediction_engine.main_engine import get_results

# Making them global so they can easily be modified
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

dataset_num = 300  # Number of Prediction Data points to send to database. Currently, the number of data points is 1000: set by limit in main_engine.get_klines function.


def generate_data() -> None:
    """Generates and saves prediction and backtest data to the database."""
    print("Generating New data")
    print("Running in Production mode. To run in dev mode, please use 'from app.prediction_engine.main_engine_local_env import get_results' instead of 'from app.prediction_engine.main_engine import get_results'")
    results, calculated_data = get_results(
        symbols_list[:3], intervals[:2], strategies[:1]
    )  # check

    # Drop existing data and create new tables so the database only contains fresh values
    main_app.db.drop_all()
    main_app.db.create_all()

    for symbol, dataset in calculated_data.items():
        now = datetime.now()
        new_prediction = main_app.Prediction(
            date_created=now,
            symbol=symbol,
            data=dataset[len(dataset) - dataset_num :].to_dict(
                "list"
            ),  # send only 300 data points to database. Most Recent
        )
        main_app.db.session.add(new_prediction)
        main_app.db.session.commit()

    for symbol_strategy, dataset in results.items():
        now = datetime.now()
        new_backtest = main_app.Backtest(
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
        main_app.db.session.add(new_backtest)
        main_app.db.session.commit()
    print("Done!")


def get_database_data(typ: str, symbol: Union[str, None] = None) -> List:
    """Returns the data in database whose table name == typ. If symbol is provided, it returns only the column in the data base whose symbol property == symbol"""
    return_list = []
    if typ == "Prediction":
        print("Getting Predictions")
        predictions = main_app.Prediction.query.all()
        for prediction in predictions:
            if (prediction.symbol == symbol) or (not symbol):
                return_list.append(
                    {
                        "id": prediction.id,
                        "symbol": prediction.symbol,
                        "data": prediction.data,
                        "date_created": prediction.date_created.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )
    else:
        print("Getting Backtests")
        backtests = main_app.Backtest.query.all()
        for backtest in backtests:
            if (backtest.symbol_strategy == symbol) or (not symbol):
                return_list.append(
                    {
                        "id": backtest.id,
                        "date_created": backtest.date_created.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "symbol": backtest.symbol,
                        "symbol_strategy": backtest.symbol_strategy,
                        "strategy": backtest.strategy,
                        "start_date": backtest.start_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "end_date": backtest.end_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
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

    return return_list


def get_minutes_to_nearest_hr() -> int:
    """Returns the number of minutes to left to get to the nearest 1 hr. It always adds an extra 1 minute"""
    current_time = datetime.now()
    current_minute = current_time.minute
    minutes_to_nearest_hour = 1 if (60 - current_minute) > 55 else 61 - current_minute
    return minutes_to_nearest_hour
