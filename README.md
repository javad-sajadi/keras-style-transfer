# Neural Style Transfer with Keras & VGG19

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An implementation of **Neural Style Transfer** (Gatys et al., 2015) in Keras / TensorFlow, synthesizing the semantic content of a base photograph with the artistic texture and colors of a painting using pre-trained VGG-19 deep feature representations.

---

## 🎨 Overview & Mathematical Formulation

Neural Style Transfer optimizes an initially noisy or content canvas $G$ to minimize a joint loss function:

$$\mathcal{L}_{\mathrm{total}}(G) = \alpha \, \mathcal{L}_{\mathrm{content}}(C, G) + \beta \, \mathcal{L}_{\mathrm{style}}(S, G)$$

### 1. Content Representation
Feature activations from deep layer `conv4_2` capture high-level semantic layout without fine pixel constraints:
$$\mathcal{L}_{\mathrm{content}}(C, G) = \frac{1}{4 n_H n_W n_C} \sum_{i,j,k} \left(a^{[l] (C)}_{i,j,k} - a^{[l] (G)}_{i,j,k}\right)^2$$

### 2. Style Representation (Gram Matrices)
Artistic style is modeled by cross-channel correlations (Gram matrices $\mathcal{G}^{[l]}$) across multiple hierarchical layers (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`):
$$\mathcal{G}_{i, j}^{[l]} = \sum_{k} a_{i, k}^{[l]} \, a_{j, k}^{[l]}$$

---

## 🖼️ Visual Demo

| Content Image | Style Painting | Generated Stylized Synthesis |
| :---: | :---: | :---: |
| ![Content](demo/images/louvre_small.jpg) | ![Style](demo/images/monet.jpg) | ![Generated](demo/outputs/generated_image.jpg) |
| *Louvre Museum* | *Claude Monet* | *Neural Synthesis* |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/javad-sajadi/keras-style-transfer.git
cd keras-style-transfer
pip install -r requirements.txt
```

### 2. Download Pre-trained VGG-19 Weights
Download the pre-trained `imagenet-vgg-verydeep-19.mat` weights file:
```bash
python -c "from keras_style_transfer.library.download_utils import download_vgg19; download_vgg19('demo/pretrained-model')"
```

### 3. Run Style Transfer
```bash
cd demo
python style_transfer.py
```

---

## 📁 Repository Structure

```
.
├── demo/
│   ├── style_transfer.py         # Modernized execution script with PIL integration
│   ├── images/                   # Sample content & style images
│   └── outputs/                  # High-resolution generated stylized outputs
├── keras_style_transfer/
│   └── library/
│       ├── style_transfer.py     # Core optimization engine & loss definitions
│       ├── nst_utils.py          # Modernized VGG-19 matrix loading & PIL image saving
│       └── download_utils.py     # Automated weight download utilities
├── requirements.txt              # Standard Python dependencies
└── .gitignore                    # Clean Git ignore rules for checkpoints and large weights
```

---

## 📜 Attribution & License
- Implemented and modernized by [Javad Sajadi](https://github.com/javad-sajadi).
- Original NST algorithm by Gatys, Ecker & Bethge (2015).
- Released under the [MIT License](LICENSE).
