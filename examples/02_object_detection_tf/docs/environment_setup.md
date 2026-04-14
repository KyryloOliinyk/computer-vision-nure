# ⚙️ Environment Setup

## 1. Clone the Repository

```bash
git clone https://github.com/CyrilOleynik/ObjectDetectionTF.git
cd ObjectDetectionTF
```

## 2. Create Conda Environment

Make sure [Conda](https://www.anaconda.com/download/) is installed.
Refer to [the docs](https://docs.conda.io/en/latest/) section if you need help.


```bash 
conda env create -f environment.yml
conda activate ObjectDetection
```

This installs all required packages including TensorFlow, Protobuf, etc.

## 3. Install TensorFlow Object Detection API


📚 **Official Manual**: [TensorFlow Object Detection API Installation Guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-install)

- Clone the official TensorFlow models repository:

```bash
git clone https://github.com/tensorflow/models.git
```

- Create proto files and manually install the library:

```bash
cd models/research/
# Compile proto files
protoc object_detection/protos/*.proto --python_out=.
# Install the Object Detection API
cp object_detection/packages/tf2/setup.py .
python -m pip install . --no-deps
```

> ℹ️ **NOTE:** TensorFlow Object Detection API is not a part of TensorFlow library and needs to be installed separately.
> Also dependencies manually installed in the environment.yml to avoid version incompatibilities.

- Verify installation:

```bash
# From directory models/research/
python object_detection/builders/model_builder_tf2_test.py
```

---
