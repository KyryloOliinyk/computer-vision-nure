from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils

from inference_config import InferenceConfig


class ObjectDetector:
    def __init__(self, config: InferenceConfig):
        self.model = self._load_saved_model(config.model_path)
        self.category_index = self._load_label_map(config.label_map_path)
        self.threshold = config.threshold

    def _load_saved_model(self, model_path: Path):
        print(f"Loading model from {model_path}")
        model_path /= 'saved_model'
        model = tf.saved_model.load(model_path)
        print("Model loaded")
        return model

    def _load_label_map(self, label_map_path: Path) -> Dict:
        return label_map_util.create_category_index_from_labelmap(label_map_path)

    def detect(self, image_np: np.ndarray) -> Dict:
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # Also model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
        raw_detections = self.model(input_tensor)

        num = int(raw_detections.pop('num_detections'))
        detections = {
            k: v[0, :num].numpy() for k, v in raw_detections.items()
        }
        detections['num_detections'] = num
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections

    def annotate(self, image_np: np.ndarray, detections: Dict) -> np.ndarray:
        return visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=50,
            min_score_thresh=self.threshold,
            agnostic_mode=False
        )
