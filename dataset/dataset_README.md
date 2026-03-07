# 🧠 Brain Tumor MRI Classification

A curated dataset of **7,023 MRI scans** across four classes for training tumor detection and classification models.

---

## Classes

| Class | Training | Testing | Description |
|---|---|---|---|
| `glioma` | 1,321 | 300 | Malignant brain/spinal cord tumors |
| `meningioma` | 1,339 | 306 | Tumors of the meninges (usually benign) |
| `pituitary` | 1,457 | 300 | Tumors at the base of the brain |
| `notumor` | 1,595 | 405 | Healthy baseline scans |
| **Total** | **5,712** | **1,311** | **7,023 images** |

---

## Setup

### 1. Download

Download from Kaggle (free account required):

```
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

Or via the Kaggle CLI:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d dataset/
```

### 2. Expected Directory Structure

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## Dataset Details

- **Format:** JPG images at varying native resolutions
- **Preprocessing:** Resize to `128×128` during training (configurable)
- **Split:** Pre-split into `Training/` and `Testing/` — do not shuffle across splits to preserve benchmark integrity
- **Source:** [Masoud Nickparvar on Kaggle](https://www.kaggle.com/masoudnickparvar)
- **License:** Check Kaggle listing before use in production or publication

---

## Attribution

```
Nickparvar, M. (2021). Brain Tumor MRI Dataset.
Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```