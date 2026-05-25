import numpy as np

class WindowDataGenerator:
    def __init__(self, scaled_data: np.ndarray, window_size: int):
        self.scaled_data = scaled_data
        self.window_size = window_size

    def generate(self):
        x, y = [], []
        for i in range(self.window_size, len(self.scaled_data)):
            x.append(self.scaled_data[i - self.window_size:i, 0])
            y.append(self.scaled_data[i, 0])
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        return x, y
