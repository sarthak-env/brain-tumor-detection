# 🧠 NeuroScan AI — Brain Tumor Classification

A deep learning system that classifies brain MRI scans into 4 categories using **VGG16 Transfer Learning**. Built on top of ImageNet pretrained weights with a custom classification head fine-tuned for medical imaging.

---

## Results

| Metric | Score |
|--------|-------|
| Model | VGG16 (Transfer Learning) |
| Input Size | 224 × 224 |
| Total Classes | 4 |
| Training Images | 5,712 |
| Test Images | 1,311 |
| Base Model | VGG16 (ImageNet weights) |
| Fine-tuned Layers | Last 3 Conv layers |

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
brain-tumor-detection/
├── frontend/       → Web interface (HTML/CSS/JS)
├── backend/        → Flask API serving the model
├── notebook/       → Training notebook (Google Colab)
└── dataset/        → Dataset download instructions
```

---

## How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
Open `frontend/index.html` in any browser.

---

## Dataset

Brain Tumor MRI Dataset by Masoud Nickparvar
- Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 7,023 MRI images across 4 classes
- Pre-split into Training (5,712) and Testing (1,311)

---

## Model Architecture

```
Input (224×224×3)
      ↓
VGG16 Base (ImageNet weights)
  - 13 Convolutional layers
  - Last 3 layers fine-tuned
  - All others frozen
      ↓
Flatten
      ↓
Dropout(0.3)
      ↓
Dense(256, relu)
      ↓
Dropout(0.2)
      ↓
Dense(4, softmax) → Prediction
```

---

## Why Transfer Learning?

Training a deep CNN from scratch on medical imaging data is difficult — we typically don't have millions of labelled MRI scans, and training converges very slowly without a strong starting point.

VGG16 was originally trained on ImageNet, a dataset of 1.4 million images across 1000 categories. It learned a rich hierarchy of visual features — edges and textures in early layers, shapes and patterns in middle layers, and high-level structures in deeper layers. These low-level features are universal and transfer well to MRI scans.

Instead of initialising with random weights, we start with VGG16's pretrained weights and only adapt the final layers to our specific task — classifying brain tumours into four categories. We freeze most of the base model and only unfreeze the last 3 convolutional layers for fine-tuning. On top we attach a custom classification head outputting probabilities for 4 classes.

**Result:** faster convergence, better accuracy on a small dataset, and far less compute compared to training from scratch.

---

## Tech Stack

- **Model:** VGG16 + TensorFlow 2.x / Keras
- **Backend:** Flask + Flask-CORS
- **Frontend:** HTML, CSS, JavaScript, Lucide Icons
- **Training:** Google Colab (NVIDIA Tesla T4 GPU)
- **Dataset:** Kaggle — Brain Tumor MRI Dataset

---

## Class Order (from training)

```python
{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
```

---

*Built as a college deep learning project.*
