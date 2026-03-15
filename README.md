# NeuroScan AI — Brain Tumor Classification

A deep learning system that classifies brain MRI scans into 4 categories using a Convolutional Neural Network built completely from scratch. No transfer learning. No pretrained weights.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 89% |
| Macro F1-Score | 87% |
| Total Classes | 4 |
| Training Images | 5,712 |
| Test Images | 1,311 |
| Total Parameters | 8,516,420 |

---

## Classes

| Class | Description | Risk Level |
|-------|-------------|------------|
| Glioma | Tumor in the glial cells of the brain | High |
| Meningioma | Tumor in the membrane surrounding the brain | Moderate |
| No Tumor | Healthy brain, no abnormalities detected | None |
| Pituitary | Tumor in the pituitary gland | Low–Moderate |

---

## Project Structure

```
neuroscan-ai/
├── frontend/       → Web interface
├── backend/        →
├── notebook/       → Training notebook (Google Colab)
└── dataset/        → Dataset download instructions
```

---

## Dataset

Brain Tumor MRI Dataset by Masoud Nickparvar
- Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 7,023 MRI images across 4 classes
- Pre-split into Training (5,712) and Testing (1,311)

---

## Model Architecture

```
Input (128×128×3)
      ↓
Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
      ↓
Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
      ↓
Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
      ↓
Flatten → Dense(256) → Dropout(0.3)
      ↓
Dense(128) → Dropout(0.2)
      ↓
Dense(4) + Softmax → Prediction
```

---

## Tech Stack

- **Model:** TensorFlow 2.19 / Keras
- **Backend:** 
- **Frontend:** HTML, CSS, JavaScript, Lucide Icons
- **Training:** Google Colab (NVIDIA Tesla T4 GPU)
- **Dataset:** Kaggle — Brain Tumor MRI Dataset

---

## Classification Report

```
              precision    recall  f1-score   support

      glioma       0.83      0.98      0.89       300
  meningioma       0.97      0.56      0.71       306
     notumor       0.88      1.00      0.94       405
   pituitary       0.92      0.99      0.96       300

    accuracy                           0.89      1311
   macro avg       0.90      0.88      0.87      1311
weighted avg       0.90      0.89      0.88      1311
```

---

## Why CNN From Scratch?

The goal was to deeply understand how Convolutional Neural Networks work by building every layer manually — Conv2D, BatchNormalization, MaxPooling, Dropout, Dense — without relying on pretrained weights. Transfer learning with VGG16 or ResNet would achieve 95%+ but provides less understanding of the underlying architecture.

---

*Built as a college deep learning project.*
