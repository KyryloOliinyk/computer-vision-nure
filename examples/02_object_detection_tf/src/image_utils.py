from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ImageUtils:
    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))

    @staticmethod
    def show_image(image_np: np.ndarray, title: str = ""):
        matplotlib.use('TkAgg')
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.axis('off')
        plt.imshow(image_np)
        plt.show()

    @staticmethod
    def get_images(directory: Path) -> List[Path]:
        return sorted([f for f in directory.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
