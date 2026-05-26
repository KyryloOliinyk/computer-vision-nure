import datetime
import keras
import numpy as np

from pathlib import Path
from typing import Tuple, List, Optional


class ExchangeRateModelLSTM:
    def __init__(self):
        self.model = None

    @staticmethod
    def _create_model_architecture(input_shape: Tuple[int, int]) -> keras.Sequential:
        return keras.Sequential([
            keras.Input(shape=input_shape, name="window_data"),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])

    def compile_model(
            self,
            input_shape: Tuple[int, int],
            learning_rate: float = 1e-3,
            plot_architecture: bool = True,
            show_summary: bool = True) -> None:
        if self.model is None:
            self.model = self._create_model_architecture(input_shape)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )

        if show_summary:
            self.model.summary()

        if plot_architecture:
            self._plot_model_architecture()

    def train_model(
            self,
            train_features: np.ndarray,
            train_targets: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray],
            epochs: int = 100,
            batch_size: int = 16,
            enable_tensorboard: bool = True,
            tensorboard_log_dir: Optional[Path] = None
    ) -> keras.callbacks.History:
        return self.model.fit(
            train_features,
            train_targets,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self._create_callbacks(enable_tensorboard, tensorboard_log_dir),
            validation_data=validation_data
        )

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        return self.model.predict(test_data)

    def evaluate_accuracy(self, test_features: np.ndarray, test_target: np.ndarray) -> float:
        return self.model.evaluate(test_features, test_target, verbose=2)

    def _plot_model_architecture(self) -> None:
        keras.utils.plot_model(
            self.model,
            "../docs/model_architecture.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
        )

    @staticmethod
    def _create_callbacks(
            enable_tensorboard: bool,
            tensorboard_log_dir: Optional[Path]
    ) -> List[keras.callbacks.Callback]:
        callbacks: List[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

        if enable_tensorboard:
            log_dir = tensorboard_log_dir or f"../logs/train/{datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')}"
            callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

        return callbacks
