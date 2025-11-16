# Lecture 10: Object Recognition

## Objectives
- Understand and be able to describe fundamental workflows of object recognition  
- Compare traditional vs. modern approaches  
- Learn features, HOG, neural networks, CNNs, training workflow  

## Introduction
Object recognition = identifying objects in images using patterns + features.

### Workflow
1. Input acquisition  
2. Pre-processing  
3. Feature extraction  
4. Recognition  
5. Output  

## Feature Engineering
- To select transform **Raw Data** -> **Suitable Data** for ML algorithm (vector)
- Simpler Expressor but its need amount of data to recognize it, here are some :
  - Colour (RGB, HSV, CIELAB, YCbCr)  
  - Texture (Co-occurrence matrix, Tamura features)  
  - Shape (region properties)  
  - Edges (Sobel, Prewitt, Canny)

## HOG (Histogram of Oriented Gradients)
- Powerful feature based on gradients (edges) of images (Gradient-based descriptor)
- Steps:
  1. Color normalization (Gamma Equalization), (optional) 
  2. Gradient computation (Get Edge & Direction)
    - Gx = [-1 0 1] 
    - Gy = [-1 0 1]^T 
    - Direction = $tan^-1(\frac{Gy}{Gx})$
    - Magnitude = $\sqrt{Gx^2 + Gy^2}$
  3. Cell histograms : Partition to non-overlapping cell -> calculate HOG
    - Weigted Vote
      - Pixel D  
        - Angle = **26.6°** (between 20° and 40°)  
        - Magnitude = **8.94**
        - Distances  
          - to 20°: 6.6  
          - to 40°: 13.4  
          - interval = 20
        - Weights
          - w_20° = 13.4 / 20 = 0.67 : weight for 20 
          - w_40° = 6.6 / 20 = 0.33 : weight for 40
        - 20° bin → 8.94 × 0.67 = 5.99
        - 40° bin → 8.94 × 0.33 = 2.95
  4. Construct overlap block  
  5. Concatenate Block → feature vector = #block * block_size * #bin  

## Neural Networks
Perceptron: `y = f(Wx + b)`  
Activation (initially used a binary threshold):
  - ReLU : f(x) = max(x,0) 
Perceptron -> Multilayer Perceptron (MLP) is most basic Neural Network

## Training Workflow
- Train/Validation/Test split  
  - Train : 60-80%, Adjust internal params (model) to minimize error
  - Validation : 10-20%, Fine-Tune hyperparams (external : learning rate, epoch)
    - Batch : One forward pass + one backpropagation on a subset of the training data.
    - Epoch : Using the entire training dataset once (many batches inside one epoch).
    - Learning Rate : How big each weight-update step is during training
  - Test : Simulate Real-World (Fully Unseen)
- Loss: How well a model is performing
  - Mean Squared Error (MSE): Used for regression problems (predicting continuous values).
  - Binary Cross Entropy (BCE): Used for binary classification (2 classes or sigmoid output).
  - Categorical Cross Entropy (CCE): Used for multi-class classification with softmax output.
- Optimizers: Algorithm that uses the gradients from backpropagation to update the weights
  - Backpropagation computes the gradients.  
Optimizer uses those gradients to update the weights efficiently.
  - Gradient Descent (GD)
    - Pros: Stable, moves directly toward minimum.
    - What it does: Uses the entire dataset (1 epoch) to compute gradients each update.
    - When to use: Small datasets where full-batch updates are affordable.

  - Stochastic Gradient Descent (SGD)
      - Pros: Fast updates, handles large datasets, introduces useful randomness.
      - What it does: Updates weights using one sample (batch) at a time.
      - When to use: Large datasets, online/streaming training, general training baseline.

  - Momentum
      - Pros: Reduces oscillation, speeds up training, smoother convergence.
      - What it does: Adds a velocity term that accumulates past gradients.
      - When to use: Loss surface with ravines or zig-zagging gradients.

  - RMSprop
      - Pros: Adaptive learning rate per parameter, works well for noisy gradients.
      - What it does: Divides gradient by an exponentially decaying average of squared gradients.
      - When to use: Recurrent networks, non-stationary or noisy problems.

  - Adam
      - Pros: Fast, stable, widely used, combines Momentum + RMSprop.
      - What it does: Tracks both gradient mean and variance to adapt learning rate.
      - When to use: Default choice for most deep learning tasks.

- Metrics: 
  - T/F, P/N : 
    - T/F บอกว่าการเดานี้ถูกหรือผิด , Correct or not
    - P/N บอกว่าเดาอะไร , What model predict
    - TP : เดาว่าใช่ และเป็นการเดาที่ถูกต้อง, correct positive
  - **X** is wildcard
  - Accuracy : Correct/All = TX/XX 
  - Precision : Correct Positive/All Predict Positive = TP/XP
    - If bias to pick Negative, Precision will look good
  - Recall : Correct Positive/All Actual Positive = TP/TP+FN
    - If bias to pick Positive, Recall will look good
  - F-score : Avoid Precision & Recall Bias  
  $$2*\frac{precision*recall}{precision+recall}$$

## CNNs (Convolutional Neural Networks)
3 Main Layers
- Convolution: Extract features by sliding filters to detect edges, textures, and patterns.
  - Depth : depend on input image depth, 3 for RGB 1 for Gray
  - Stride : #pixel when shift kernel
  - Padding : How to handle out of image pixel 
- Pooling: Downsample feature maps to reduce size and keep important information.
  - Max Pool : Choose max from area
- Fully Connected: Standard neural network layers for final classification or prediction.
  - #params = (#input * #output) + #output = (#input + 1) * #output
    -  Every input (x) connect with output -> so have weight(W) for each connection
    - Every output-node have Bias (b) so have to plus #output

## Improving Generalization
- Dropout
    - Purpose: Reduce overfitting.
    - What it does: Randomly disables a percentage of neurons during training so the model doesn’t rely too heavily on specific nodes.
      - When dropout is applied during training, the dropped neurons are disabled for **that batch only**.
      - They do NOT participate in the forward pass → they produce no output.
      - They do NOT participate in the backward pass → they receive no gradient → no weight update.
      - In the next batch, a different random set of neurons may be dropped.

- Batch Normalization (BatchNorm)
    - Purpose: Stabilize and speed up training.
    - What it does: Make output in a acceptable range with normalize distribution  
      - During training: BatchNorm computes mean and variance from each mini-batch. These values change every batch.
      - At the same time, it updates running_mean and running_variance (moving averages).
      - During inference: BatchNorm uses the running_mean and running_variance, not the batch statistics, to keep predictions stable.

- More filters/layers/epoch
- Data augmentation  

## Traditional vs Modern Approaches
**Traditional**: handcrafted features (HOG, SIFT) + SVM
**Modern**: CNN learns features automatically → higher accuracy  

