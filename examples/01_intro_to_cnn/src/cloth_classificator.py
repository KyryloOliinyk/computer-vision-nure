import datetime

import tensorflow as tf


class ClothClassificator:
    def __init__(self):
        self.model = None
        self.class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    @staticmethod
    def load_training_data():
        return tf.keras.datasets.fashion_mnist.load_data()

    @staticmethod
    def prepare_data(data):
        return (data / 255.0).reshape(-1, 28, 28, 1)

    def compile_model(self):
        if self.model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.model.summary()

    def train_model(self, train_images, train_labels, epochs=10, enable_tensorboard=False, tensorboard_log_dir=None):
        callbacks = []
        if enable_tensorboard:
            log_dir = tensorboard_log_dir or f"logs/fit/{datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')}"
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

        self.model.fit(train_images, train_labels, epochs=epochs, callbacks=callbacks)

    def predict(self, test_images):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        return probability_model.predict(test_images)

    def evaluate_accuracy(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels, verbose=2)
