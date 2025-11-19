# 🐱🐶 Cat vs Dog Image Classification using CNN

A deep learning project implementing a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow/Keras.

## 📋 Project Overview

This project uses a custom CNN architecture to perform binary image classification on the popular Kaggle Dogs vs Cats dataset. The model achieves **~93% validation accuracy** through careful architecture design, data augmentation, and training optimization techniques.

## 🎯 Features

- Custom CNN architecture with 4 convolutional blocks
- Data augmentation for improved generalization
- Batch normalization and dropout for regularization
- Learning rate reduction and early stopping callbacks
- Train/validation/test split for robust evaluation

## 🏗️ Model Architecture

```
Input (128x128x3)
    ↓
[Block 1] Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
[Block 2] Conv2D(64) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
[Block 3] Conv2D(128) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
[Block 4] Conv2D(256) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.2)
    ↓
Output Dense(1, sigmoid)
```

**Total Parameters:** 5,112,001 (19.50 MB)
- Trainable params: 5,110,017
- Non-trainable params: 1,984

## 📊 Dataset

- **Source:** [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats)
- **Total Images:** 25,000
- **Training Set:** 20,000 images (80%)
- **Validation Set:** 2,500 images (10%)
- **Test Set:** 2,500 images (10%)
- **Classes:** Binary (Cat: 0, Dog: 1)

## 🔧 Technologies Used

- **Python 3.12**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **scikit-learn** - Train-test split and metrics
- **PIL** - Image processing

## 📈 Training Configuration

### Hyperparameters
- **Image Size:** 128×128×3
- **Batch Size:** 32
- **Initial Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Epochs:** 30 (with early stopping)

### Data Augmentation
- Rotation range: ±15°
- Horizontal flip: True
- Zoom range: 0.2
- Shear range: 0.1
- Width/height shift: 0.1
- Fill mode: reflect
- Rescaling: 1/255

### Callbacks
- **ReduceLROnPlateau:** Reduces learning rate by 0.5 when validation accuracy plateaus (patience=2)
- **EarlyStopping:** Stops training if validation loss doesn't improve (patience=3)

## 🎯 Results

- **Best Validation Accuracy:** ~92.76% (Epoch 9)
- **Best Validation Loss:** ~0.1746
- **Training stopped at:** Epoch 12 (early stopping triggered)

The model shows strong performance with:
- Good generalization to unseen data
- Minimal overfitting due to regularization techniques
- Efficient convergence with adaptive learning rate

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MohitRawat017/Cat-V-Dog-CNN-Model.git
cd Cat-V-Dog-CNN-Model
```

2. Download the Kaggle dataset:
```bash
# Ensure kaggle.json is in ~/.kaggle/
kaggle competitions download -c dogs-vs-cats
```

3. Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

## 📁 Project Structure

```
Cat-V-Dog-CNN-Model/
│
├── main.ipynb          # Main training notebook
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## 💡 Key Implementation Details

### Why This Architecture Works

1. **Progressive Feature Extraction:** Each convolutional block doubles the number of filters (32→64→128→256), allowing the network to learn increasingly complex features.

2. **Regularization Strategy:**
   - Batch normalization after each convolution for stable training
   - Dropout (20%) to prevent overfitting
   - Data augmentation to increase dataset diversity

3. **Adaptive Learning:**
   - ReduceLROnPlateau automatically adjusts learning rate
   - EarlyStopping prevents unnecessary training

## 🔍 Usage Example

To train the model:
```python
# The notebook handles everything automatically
# Just run all cells in main.ipynb
```

To make predictions on new images:
```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model.load_weights('best_model.h5')  # if saved

# Prepare image
img = image.load_img('path/to/image.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print('Dog' if prediction[0][0] > 0.5 else 'Cat')
```

## 📝 Future Improvements

- [ ] Implement transfer learning (ResNet, VGG, EfficientNet)
- [ ] Add confusion matrix and classification report
- [ ] Deploy model as a web API using Flask/FastAPI
- [ ] Add model versioning and experiment tracking (MLflow)
- [ ] Implement test-time augmentation for better predictions
- [ ] Create a Streamlit/Gradio demo interface

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Mohit Rawat**
- GitHub: [@MohitRawat017](https://github.com/MohitRawat017)

## 🙏 Acknowledgments

- Kaggle for providing the Dogs vs Cats dataset
- TensorFlow/Keras team for the excellent deep learning framework
- The open-source community for inspiration and resources

---

⭐ If you find this project useful, please consider giving it a star!
