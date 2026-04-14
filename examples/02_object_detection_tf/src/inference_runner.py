from pathlib import Path

from image_utils import ImageUtils
from inference_config import InferenceConfig
from object_detector import ObjectDetector


class InferenceRunner:
    def __init__(self, images_directory: Path, config: InferenceConfig):
        self.config = config
        self.images_directory = images_directory
        self.detector = ObjectDetector(config)

    def run(self):
        image_paths = ImageUtils.get_images(self.images_directory)
        for path in image_paths:
            print(f"Processing {path}...")

            image = ImageUtils.load_image(path)
            detections = self.detector.detect(image)
            annotated = self.detector.annotate(image.copy(), detections)

            ImageUtils.show_image(annotated, title=path.name)
