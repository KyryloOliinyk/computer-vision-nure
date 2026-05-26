import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from exchange_rate_model import ExchangeRateModelLSTM as LSTMModel
from exchange_rate_dataset import ExchangeRateDataset
from window_data_generator import WindowDataGenerator


def plot_test_results(
    dates_test: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    mae: float,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dates_test, actual,    color="#1a73e8", linewidth=1.8, label="Actual")
    ax.plot(dates_test, predicted, color="#e8710a", linewidth=1.8,
            linestyle="--", label="Predicted")

    ax.fill_between(dates_test, actual, predicted,
                    alpha=0.12, color="#e8710a")

    ax.set_title("LSTM — Test Set: Actual vs Predicted USD/UAH", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate (UAH per USD)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(0.98, 0.05, f"MAE: {mae:.4f} UAH",
            transform=ax.transAxes, ha="right",
            fontsize=10, color="#333")

    plt.tight_layout()
    plt.savefig("test_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    dataset = ExchangeRateDataset("../data/usd_uah_exchange.csv")
    scaled = dataset.scale()

    raw_df = dataset.get_raw_data()
    print(f"Dataset: {len(raw_df)} days  "
          f"({raw_df.index[0].date()} → {raw_df.index[-1].date()})")

    look_back = 30
    window_data_generator = WindowDataGenerator(scaled, look_back)
    x, y = window_data_generator.generate()

    split = int(len(x) * 0.9)
    X_train, y_train = x[:split], y[:split]
    X_test, y_test = x[split:], y[split:]

    test_start_idx = split + look_back
    dates_test = raw_df.index[test_start_idx: test_start_idx + len(X_test)]

    print(f"Train samples: {len(X_train)}   Test samples: {len(X_test)}")

    model = LSTMModel()
    model.compile_model(input_shape=(look_back, 1))

    history = model.train_model(
        X_train, y_train,
        validation_data=(X_test, y_test)
    )

    preds_scaled = model.predict(X_test)
    preds_actual = dataset.inverse_scale(preds_scaled)
    actual       = dataset.inverse_scale(y_test)

    loss, mae = model.evaluate_accuracy(X_test, y_test)
    print(f"\nTest  MSE: {loss:.6f}   MAE: {mae:.4f} UAH")
    plot_test_results(dates_test, actual, preds_actual, mae)


# import numpy as np
# import matplotlib.pyplot as plt
#
# from exchange_rate_model import ExchangeRateModelLSTM as LSTMModel
# from exchange_rate_dataset import ExchangeRateDataset
# from window_data_generator import WindowDataGenerator
#
#
# def plot_predictions(actual, predicted):
#     plt.figure(figsize=(12, 6))
#     plt.plot(actual, label='Actual USD/UAH')
#     plt.plot(predicted, label='Predicted USD/UAH')
#     plt.title('LSTM Forecast of USD/UAH Exchange Rate')
#     plt.xlabel('Day')
#     plt.ylabel('Rate')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     dataset = ExchangeRateDataset("../data/usd_uah_exchange.csv")
#     scaled = dataset.scale()
#
#     look_back = 30
#     window_data_generator = WindowDataGenerator(scaled, look_back)
#     x, y = window_data_generator.generate()
#
#     split = int(len(x) * 0.9)
#     X_train, y_train = x[:split], y[:split]
#     X_test, y_test = x[split:], y[split:]
#
#     model = LSTMModel()
#     model.compile_model(input_shape=(look_back, 1))
#     model.train_model(X_train, y_train, validation_data=(X_test, y_test))
#
#     predictions = model.predict(X_test)
#     predictions_unscaled = dataset.inverse_scale(predictions)
#     y_test_unscaled = dataset.inverse_scale(y_test.reshape(-1, 1))
#
#     plot_predictions(y_test_unscaled, predictions_unscaled)
#     visualize_window(x, y, index=5)
