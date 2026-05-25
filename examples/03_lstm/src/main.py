import numpy as np
import matplotlib.pyplot as plt
from src.exchange_rate_model import ExchangeRateModelLSTM as LSTMModel
from src.exchange_rate_dataset import ExchangeRateDataset
from src.window_data_generator import WindowDataGenerator


def visualize_window(X: np.ndarray, y: np.ndarray, index: int):
    look_back = X.shape[1]
    window = X[index].flatten()
    target = y[index]

    plt.figure(figsize=(10, 4))
    plt.plot(range(look_back), window, marker='o', label='Input Window')
    plt.plot(look_back, target, 'ro', label='Target')
    plt.title('Sample LSTM Input Window and Target')
    plt.xlabel('Day')
    plt.ylabel('Scaled Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual USD/UAH')
    plt.plot(predicted, label='Predicted USD/UAH')
    plt.title('LSTM Forecast of USD/UAH Exchange Rate')
    plt.xlabel('Day')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    dataset = ExchangeRateDataset("data/usd_uah_exchange.csv")
    scaled = dataset.scale()

    look_back = 30
    window_data_generator = WindowDataGenerator(scaled, look_back)
    x, y = window_data_generator.generate()

    split = int(len(x) * 0.9)
    X_train, y_train = x[:split], y[:split]
    X_test, y_test = x[split:], y[split:]

    model = LSTMModel()
    model.compile_model(input_shape=(look_back, 1))
    model.train_model(X_train, y_train, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    predictions_unscaled = dataset.inverse_scale(predictions)
    y_test_unscaled = dataset.inverse_scale(y_test.reshape(-1, 1))

    plot_predictions(y_test_unscaled, predictions_unscaled)
    visualize_window(x, y, index=5)
