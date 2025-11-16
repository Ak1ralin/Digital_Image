# Lecture 10: Object Recognition

## Overview
Object recognition involves identifying objects in images using either **traditional** feature-engineered methods or **modern deep learning** approaches.

---

## Object Recognition Workflow
1. Input Acquisition (image sensors)
2. Preprocessing (noise removal, enhancement, segmentation)
3. Feature Extraction (handcrafted or learned)
4. Recognition (rule-based, ML, or deep learning)
5. Output (classified result)

---

## Traditional Approach
### Features
- **Color Models:** RGB, HSV, CIELAB, YCbCr  
- **Texture:** Co-occurrence matrix, Tamura features  
- **Shape:** Region properties  
- **Edge:** Sobel, Prewitt, Canny  

### HOG (Histogram of Oriented Gradients)
- Detects gradients and edges.
- Steps: color normalization → gradient calculation → orientation binning → block normalization → classifier (usually SVM).

---

## Neural Networks
- Composed of **neurons** imitating biological ones.  
- **Perceptron:** Linear model with weights and bias.  
- **Activation functions:** ReLU, Sigmoid, Gaussian for nonlinearity.  
- **Multilayer Perceptron (MLP):** Combines multiple perceptrons.

### Training Process
- Datasets split into **train**, **validation**, and **test** sets.
- **Loss functions:** MSE, Binary/Categorical Cross-Entropy.
- **Optimizers:** GD, SGD, Momentum, RMSprop, Adam.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

---

## Case Study: MNIST Handwritten Digits
- Dataset: 60k training, 10k testing.
- HOG features + MLP classifier used.
- Achieved ~89% accuracy.
- Steps: normalize → extract HOG → train → evaluate → visualize prediction.

---

## Modern Approach: Deep Learning
### Convolutional Neural Networks (CNN)
- Learn features automatically from raw pixels.
- Layers:
  - **Convolution:** Extract local features.
  - **Pooling:** Downsample.
  - **Fully Connected:** Classification.

#### Key Parameters
- Loss: Cross-Entropy
- Epoch, Batch Size, Accuracy metrics
- Frameworks: TensorFlow, Keras, PyTorch

---

## PyTorch Model Example
- Sequential & Custom `nn.Module` models with conv, pool, linear layers.
- Training includes forward, backward, loss optimization.
- Evaluation with `model.eval()` and `torch.no_grad()`.

### Improving Generalization
- **Dropout**, **Batch Normalization**, **Data Augmentation** prevent overfitting.

---

## Recognition Performance
Dependent on:
- Feature quality
- Dataset diversity
- Model architecture
- Hyperparameter tuning

---

## Modern Trends
- Models: AlexNet, ZFNet, GoogLeNet, VGG, ResNet, EfficientNet.
- Techniques: Transfer learning, data augmentation.
- Applications: Object detection and segmentation.

---

**Summary:**  
Object recognition evolved from handcrafted features (HOG, color, texture) to deep learning with CNNs that learn features automatically, improving accuracy and scalability.