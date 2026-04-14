# ✅ Model Validation

To detect motorbikes in images, run the following script:

```bash
python src/main.py \
  --model_path exported-models/ssd_resnet50_v1_fpn_640x640_transport \
  --label_map annotations/transport_label_map.pbtxt \
  --input_dir images/test \
  --threshold=0.3
```

Arguments:

* **_model_path_** — Path to your saved model
* **_label_map_** — Path to your label map file (`.pbtxt`)
* **_input_dir_** — Path to the directory containing input test images
* **_threshold_** — Score threshold for displaying detections

Detected objects should be annotated.

---

## 🖼️ Example Output

After running the script, the annotated images with detected objects, bounding boxes, and labels will be displayed.

### ** 🖼️ Input Image:**

![Input Image](Figure_1.jpg)

### ** 🖼️ Output (annotated):**

![Output (annotated)](Figure_1_annotated.png)

---
