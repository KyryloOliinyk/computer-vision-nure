import argparse
from pathlib import Path

from inference_config import InferenceConfig
from inference_runner import InferenceRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test images with a trained model.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the SavedModel directory.")
    parser.add_argument("--label_map", type=Path, required=True, help="Path to the label map (.pbtxt).")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to the directory containing test images.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for displaying detections.")

    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        label_map_path=args.label_map,
        threshold=args.threshold
    )

    runner = InferenceRunner(args.input_dir, config)
    runner.run()
