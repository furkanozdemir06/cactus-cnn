# 🌵 Aerial Cactus Detection with CNN

A computer vision project that detects whether aerial images contain **columnar cactus (Neobuxbaumia tetetzo)** using transfer learning with **ResNet50**.

## 📌 Project Overview

The project includes:

- Image dataset exploration
- Data augmentation
- Class imbalance handling
- ResNet50 transfer learning
- Fine-tuning
- Test-set prediction

## 📊 Dataset

Training set:

```text id="7ze7pm"
17,500 images

Cactus     : 13,136
No Cactus  : 4,364
```

Test set:

```text id="n9vot1"
4,000 images
```

Target:

```text id="k2pzg9"
has_cactus
```

## 🧠 CNN Model

The model uses **ResNet50 pretrained on ImageNet** with:

- Global Average Pooling
- Dense layer
- Dropout
- Sigmoid output
- Binary cross-entropy

Training also uses augmentation, class weights, early stopping, and learning-rate reduction.

## 📈 Results

```text id="mw0o5n"
Validation Accuracy ≈ 98.1%
```

The ResNet50 model is also fine-tuned with a lower learning rate.

## 🔮 Prediction

The trained model predicts cactus probabilities for the test images and generates:

```text id="46lufj"
submission.csv
```

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- ResNet50
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 🎯 Skills Demonstrated

- Computer Vision
- CNNs
- Transfer Learning
- Image Augmentation
- Fine-Tuning
- Binary Image Classification

---

Built with TensorFlow, Keras, and ResNet50. 🌵🧠
