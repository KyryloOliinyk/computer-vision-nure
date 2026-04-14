# 🎯Training Custom Object Detector

📚 **Official Manual**: [Training Custom Object Detector](
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

---

## 📊 Preparing the Dataset

1. Download the images into the `images/test` and `images/train` folders according to your dataset variant. 
   Typically, the ratio is 9:1–90% of the images are used for training, and the remaining 10% are used for testing,
   but you can choose whatever ratio suits your needs.
2. Annotate the images using **labelImg**. After annotation, you should have `.xml` files next to the images.
3. Create a Label Map based on the example `annotations/transport_label_map.pbtxt`.
4. Convert the `.xml` files to `.record` format.

```bash
python scripts/generate_tfrecord.py -x images/train -l annotations/transport_label_map.pbtxt -o annotations/train.record
python scripts/generate_tfrecord.py -x images/test -l annotations/transport_label_map.pbtxt -o annotations/test.record
```

---

## 🔍 Model Preparation

1. Download one of the pre-trained models from the [TensorFlow 2 Detection Model Zoo](
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
2. Extract it into the `pre-trained-models/` directory.

Example structure:

```
pre-trained-models/
└── ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
```

---

## 🧠 Model Training

1. Configure the Training Pipeline.
   Inside the `models` directory, create a folder for your custom model and add a `pipeline.config` file.  
   Refer to the [documentation section](
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline)
   for details.
2. Training the Model. This step takes a lot of time, so be aware — it can take hours.

```bash
python scripts/train_model.py \
  --model_dir=models/ssd_resnet50_v1_fpn_640x640_transport  \
  --num_train_steps=1500  \
  --pipeline_config_path=models/ssd_resnet50_v1_fpn_640x640_transport/pipeline.config  \
  --alsologtostderr
```

---

## 📦 Exporting a Trained Model

After training is complete, the next step is to export the trained model for inference.  
This exported model will be used to perform object detection. This can be done as follows:

```bash
python scripts/model_exporter.py \
  --input_type image_tensor \
  --pipeline_config_path models/ssd_resnet50_v1_fpn_640x640_transport/pipeline.config \
  --trained_checkpoint_dir models/ssd_resnet50_v1_fpn_640x640_transport/ \
  --output_directory exported-models/ssd_resnet50_v1_fpn_640x640_transport
```

---
