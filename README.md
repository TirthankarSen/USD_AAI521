# 🗑️ Smart Waste Classifier

**Deep Learning Pipeline for Automated Waste Classification**

A comprehensive comparative analysis of CNN architectures for automated waste sorting, implementing Custom CNN, MobileNetV2, and EfficientNetB0 with transfer learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-yellow.svg)](https://colab.research.google.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a complete deep learning pipeline for classifying waste images into 12 categories. The goal is to evaluate different CNN approaches for potential deployment in automated recycling facilities.

### Research Questions

1. Can transfer learning outperform a custom CNN trained from scratch?
2. Which pre-trained architecture is best suited for waste classification?
3. What are the practical considerations for deploying such systems?

### Key Findings

- **Transfer learning significantly outperforms training from scratch** (21.5 percentage point improvement)
- **MobileNetV2** achieves the best balance of accuracy (91.52%) and efficiency
- **Preprocessing compatibility** is critical—EfficientNetB0 failed due to input normalization mismatch
- **Visually similar materials** (glass types, plastic, metal) remain challenging

---

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **MobileNetV2** | **91.52%** | **88.12%** | **87.09%** | **86.82%** |
| Custom CNN | 70.00% | 67.58% | 65.72% | 64.64% |
| EfficientNetB0 | 6.81% | 5.01% | 11.44% | 3.17% |

### Per-Class Performance (MobileNetV2)

| Best Performers | F1-Score | Worst Performers | F1-Score |
|-----------------|----------|------------------|----------|
| Clothes | 98.91% | White-glass | 60.08% |
| Shoes | 96.89% | Plastic | 75.62% |
| Biological | 95.50% | Metal | 77.59% |

---

## Dataset

**Kaggle Garbage Classification Dataset** ([Link](https://www.kaggle.com/datasets/mostafaabla/garbage-classification))

- **Total Images:** 15,515
- **Classes:** 12
- **Image Format:** RGB, resized to 224×224
- **Split:** 80% training (12,415) / 20% validation (3,100)

### Class Distribution

| Category | Images | Percentage |
|----------|--------|------------|
| Clothes | 5,325 | 34.32% |
| Shoes | 1,977 | 12.74% |
| Paper | 1,050 | 6.77% |
| Biological | 985 | 6.35% |
| Battery | 945 | 6.09% |
| Cardboard | 891 | 5.74% |
| Plastic | 865 | 5.58% |
| White-glass | 775 | 5.00% |
| Metal | 769 | 4.96% |
| Trash | 697 | 4.49% |
| Green-glass | 629 | 4.05% |
| Brown-glass | 607 | 3.91% |

**Imbalance Ratio:** 8.77:1 (clothes vs. brown-glass)

---

## Model Architectures

### 1. Custom CNN (~27M parameters)

A 4-block CNN designed from scratch with progressive filter sizes:

```
Input (224×224×3)
    ↓
[Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)] × 1
    ↓
[Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)] × 1
    ↓
[Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.40)] × 1
    ↓
[Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.40)] × 1
    ↓
Flatten → Dense(512) → BN → Dropout(0.5)
    ↓
Dense(256) → BN → Dropout(0.5)
    ↓
Dense(12, softmax)
```

### 2. MobileNetV2 + Transfer Learning (~3.05M parameters, 794K trainable)

```
Input (224×224×3)
    ↓
MobileNetV2 Base (frozen, pre-trained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dense(512) → Dropout(0.5)
    ↓
BatchNormalization → Dense(256) → Dropout(0.5)
    ↓
Dense(12, softmax)
```

### 3. EfficientNetB0 + Transfer Learning (~4.05M parameters, 794K trainable)

Same custom head as MobileNetV2, with EfficientNetB0 as the frozen base.

> ⚠️ **Note:** EfficientNetB0 requires specific preprocessing (`tf.keras.applications.efficientnet.preprocess_input`) that differs from standard [0,1] normalization. Using incorrect preprocessing causes convergence failure.

---

## Installation

### Prerequisites

- Python 3.8+
- GPU recommended (NVIDIA T4 or better)
- ~10GB disk space for dataset and models

### Option 1: Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run all cells

### Option 2: Local Installation

```bash
# Clone or download the notebook
git clone <repository-url>
cd smart-waste-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install tensorflow==2.13.0 \
            numpy pandas matplotlib seaborn \
            scikit-learn opencv-python kaggle
```

### Kaggle API Setup

```python
# Create ~/.kaggle/kaggle.json with your credentials:
{
    "username": "your_kaggle_username",
    "key": "your_kaggle_api_key"
}

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage

### Quick Start

```python
# Run the entire pipeline
# 1. Open Smart_Waste_Classifier_v4.ipynb
# 2. Update Kaggle credentials in Section 1
# 3. Run All Cells
```

### Using a Trained Model for Inference

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the best model
model = tf.keras.models.load_model('results/models/MobileNetV2_best.h5')

# Class names
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 
               'clothes', 'green-glass', 'metal', 'paper', 
               'plastic', 'shoes', 'trash', 'white-glass']

# Predict on new image
def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence

# Example
label, conf = predict_waste('test_image.jpg')
print(f"Prediction: {label} ({conf:.2f}%)")
```

---

## Project Structure

```
smart-waste-classifier/
│
├── Smart_Waste_Classifier_v4.ipynb    # Main notebook
├── README.md                           # This file
│
├── results/
│   ├── models/
│   │   ├── Custom_CNN_best.h5         # Best custom CNN weights
│   │   ├── MobileNetV2_best.h5        # Best MobileNetV2 weights
│   │   └── EfficientNetB0_best.h5     # Best EfficientNetB0 weights
│   │
│   ├── figures/
│   │   ├── Custom_CNN_history.png     # Training curves
│   │   ├── MobileNetV2_history.png
│   │   ├── EfficientNetB0_history.png
│   │   ├── confusion_matrix.png
│   │   └── model_comparison.png
│   │
│   └── logs/
│       ├── Custom_CNN_log.csv         # Training logs
│       ├── MobileNetV2_log.csv
│       └── EfficientNetB0_log.csv
│
└── garbage-classification/             # Dataset (downloaded)
    ├── battery/
    ├── biological/
    ├── brown-glass/
    └── ... (12 class folders)
```

---

## Configuration

All hyperparameters are centralized in the `CONFIG` dictionary:

```python
CONFIG = {
    'DATASET_NAME': 'mostafaabla/garbage-classification',
    'DATA_DIR': '/content/garbage-classification',
    'NUM_CLASSES': 12,
    'IMAGE_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    'EPOCHS': 15,
    'INITIAL_LR': 0.001,
    'VAL_SPLIT': 0.2,
    'EARLY_STOP_PAT': 8,    # Early stopping patience
    'LR_PATIENCE': 4,        # Learning rate reduction patience
    'SEED': 42,
}
```

### Data Augmentation

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

### Training Callbacks

| Callback | Purpose | Configuration |
|----------|---------|---------------|
| EarlyStopping | Prevent overfitting | patience=8, restore_best_weights=True |
| ReduceLROnPlateau | Adaptive learning rate | factor=0.5, patience=4, min_lr=1e-7 |
| ModelCheckpoint | Save best model | monitor='val_accuracy' |
| CSVLogger | Log training history | Per-epoch metrics |

---

## Results

### Training Dynamics

- **MobileNetV2:** Rapid convergence, ~90% accuracy from epoch 1
- **Custom CNN:** Gradual improvement, moderate overfitting
- **EfficientNetB0:** Failed to converge (preprocessing mismatch)

### Confusion Patterns

**High Confusion Pairs:**
- White-glass ↔ Green-glass ↔ Brown-glass (similar shapes/materials)
- Plastic ↔ Metal (reflective surfaces)

**Critical Success:**
- Battery detection: 99.40% precision (important for hazardous waste handling)

---

## Lessons Learned

### 1. Transfer Learning is Powerful
Pre-trained ImageNet features provide a massive head start, especially with limited domain-specific data.

### 2. Preprocessing Matters
EfficientNetB0's failure demonstrates that each architecture has specific input requirements. Always use the matching preprocessing function.

### 3. Class Imbalance Requires Attention
Using balanced class weights prevented the model from ignoring minority classes like brown-glass and green-glass.

### 4. Architecture Efficiency
MobileNetV2 proves that you don't need the largest model—efficient architectures can achieve excellent results while remaining deployable on edge devices.

---

## Future Work

- [ ] **Realistic datasets:** Test with cluttered, dirty, and occluded images
- [ ] **Object detection:** Implement YOLO/Faster R-CNN for multi-item scenes
- [ ] **Ensemble methods:** Combine models for improved robustness
- [ ] **Edge deployment:** Benchmark on Raspberry Pi / Jetson Nano
- [ ] **Fix EfficientNetB0:** Apply correct preprocessing and retrain
- [ ] **Fine-tuning:** Unfreeze and fine-tune top layers of transfer learning models

---

## Authors

- **Tirthankar Sen**
- **Kesavan Rangaswamy**

Course: USD AA1-521, Group 14

---

## Acknowledgments

- **Dataset:** [Mostafa Abla](https://www.kaggle.com/mostafaabla) - Kaggle Garbage Classification
- **Frameworks:** TensorFlow, Keras, scikit-learn
- **Pre-trained Models:** ImageNet weights via `tf.keras.applications`
- **Compute:** Google Colab GPU runtime

---

## License

This project is for educational purposes as part of USD AA1-521 coursework.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sen2024wasteclassification,
  title={Deep Learning for Automated Waste Classification: A Comparative Analysis of CNN Architectures Using Transfer Learning},
  author={Sen, Tirthankar and Rangaswamy, Kesavan},
  year={2024},
  institution={University of San Diego}
}
```
