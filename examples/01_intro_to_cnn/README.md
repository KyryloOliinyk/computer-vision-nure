![Python](https://img.shields.io/badge/python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)

# Introduction into CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow
to classify clothing images from the Fashion MNIST dataset.

It is a simple educational project demonstrating:
- Image preprocessing
- CNN model building
- Training and evaluation
- Visualization of predictions

---

## Objective

The goal of this lab is to:

- Understand the basics of image classification
- Learn how Convolutional Neural Networks (CNNs) work
- Gain practical experience with TensorFlow
- Perform training, evaluation, and visualization of a deep learning model

---

## Dataset Variants for Classification

| Variant | Dataset Name         |
|---------|----------------------|
| 1       | eurosat              |
| 2       | emnist               |
| 3       | cifar-10             |
| 4       | cifar-100            | 
| 5       | colorectal_histology |
| 6       | oxford_flowers102    |
| 7       | horses_or_humans     |

---

## Dataset

We use the **Fashion MNIST** dataset:

🔗 https://www.tensorflow.org/datasets/catalog/fashion_mnist  

Dataset properties:
- 70,000 grayscale images (28x28)
- 10 classes of clothing items
- Training set: 60,000 images
- Test set: 10,000 images

---

## Model Architecture

The CNN consists of:
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D
- Flatten
- Dense (128, ReLU)
- Dense (10, Softmax)

| Layer (type)                    | Output Shape       | Param #    |
|---------------------------------|--------------------|------------|
| conv2d (Conv2D)                 | (None, 26, 26, 32) | 320        |
| max_pooling2d (MaxPooling2D)    | (None, 13, 13, 32) | 0          |
| conv2d_1 (Conv2D)               | (None, 11, 11, 64) | 18,496     |
| max_pooling2d_1 (MaxPooling2D)  | (None, 5, 5, 64)   | 0          |
| flatten (Flatten)               | (None, 1600)       | 0          |
| dense (Dense)                   | (None, 128)        | 204,928    |
| dense_1 (Dense)                 | (None, 10)         | 1,290      |

 Total params: 225,034 (879.04 KB) \
 Trainable params: 225,034 (879.04 KB) \
 Non-trainable params: 0 (0.00 B)

---

## Requirements

Create python interpreter using GUI or CLI

### Setting Up Python Virtual Environment

To run this project safely, it is recommended to use a **Python virtual environment**.

#### Using CLI (Command Line)

1. **Navigate to your project folder**:
```bash
cd examples/01_intro_to_cnn
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
```

3. Activate the virtual environment:
On Windows (CMD):
```cmd
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r examples/01_intro_to_cnn/requirements.txt
```

---

##  How to Run

Mak sure your working directory *examples/01_intro_to_cnn*. Run the main script:
```bash
python src/main.py
```
---

## Workflow

The program performs the following steps:

1. Load dataset (Fashion MNIST)
2. Normalize pixel values to [0, 1]
3. Reshape images to (28, 28, 1)
4. Build CNN model
5. Compile model (Adam optimizer, Crossentropy loss)
6. Train model
7. Evaluate on test dataset
8. Visualize predictions and confidence scores

---

## Results

The program outputs:
* Model summary
* Training process logs
* Test accuracy and loss
* Visualization of:
  * Input images
  * Predicted labels
  *  Prediction probabilities

---

## TensorBoard (Optional)

To enable TensorBoard, set:
```python
enable_tensorboard=True
```

Run TensorBoard:
```bash
tensorboard --logdir logs/fit
```

Open in browser provided url

---

## 📚 Useful reading:  
🔗 https://www.tensorflow.org/tutorials/images/classification  
🔗 https://www.tensorflow.org/tutorials/images/cnn  
🔗 https://cs231n.github.io/convolutional-networks/  
🔗 https://www.tensorflow.org/tensorboard  
🔗 https://developers.google.com/machine-learning/crash-course