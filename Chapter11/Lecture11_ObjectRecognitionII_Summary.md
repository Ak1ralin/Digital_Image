# Lecture 11: Object Recognition II

## Objectives
- Learn how to improve model accuracy using **data augmentation**.
- Understand **deep learning evolution** and **transfer learning** with pre-trained models.

---

## 1. Image Transformation and Augmentation
- **Pre-processing**: resize and convert images to tensors.
- **Augmentation**: artificially expand dataset to prevent overfitting.
  - Techniques: flip, scale, add noise, rotate, crop, translate.
- Implemented via `torchvision.transforms` in PyTorch.
- Example: Random flips and rotations on MNIST dataset.

```python
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation((-30,30)),
    transforms.ToTensor(),
])
```

---

## 2. Visualizing Filters and Feature Maps
- Filters show **what features CNN learns** (edges, textures, etc.).
- Feature maps reveal **activation patterns** across layers.
- Reference: Zeiler & Fergus (2014), ECCV.

---

## 3. Evolution of Deep Learning
| Year | Model | Key Features | Top-5 Error |
|------|--------|---------------|--------------|
| 1998 | LeNet | Digit recognition (7 layers) | — |
| 2012 | AlexNet | ReLU, Dropout, Data Augmentation | 15.3% |
| 2014 | VGGNet | 16/19 conv layers, uniform 3×3 filters | — |
| 2014 | GoogleNet | Inception modules, 1×1 convs, 22 layers | 6.7% |
| 2015 | ResNet | Skip connections, residual blocks | 3.57% |
| 2017 | ResNeXt | Split-transform-aggregate, cardinality | 5.3% |
| 2020 | EfficientNet | Compound scaling (width, depth, res) | 4.04% |
| 2021 | Vision Transformer (ViT) | Patch embedding + attention | — |

### Notes:
- CNNs evolved from local receptive fields to **global attention mechanisms** (ViTs).
- Modern ViTs replace convolution with **self-attention** and **positional embeddings**.

---

## 4. Vision Transformer (ViT)
- Splits image into patches (e.g., 224×224 → 14×14 patches).
- Adds positional embedding for spatial info.
- Uses **multi-head self-attention**, **layer normalization**, and **residual connections**.
- Class token used for final classification.
- Reference: *Vaswani et al., “Attention Is All You Need”, 2021*.

---

## 5. Variants of ViT
- **DPT (Deformable Patch Transformer)**: adaptive patch splitting.
- **Swin Transformer**: shifted windows for local-global attention.

---

## 6. CLIP: Contrastive Language–Image Pretraining (OpenAI, 2021)
- Jointly trains image + text encoders on 400M image-text pairs.
- Learns to match images and captions via contrastive loss.
- Enables **zero-shot classification** with text prompts.
- Competitive with supervised ResNet-50 on ImageNet.

---

## 7. Transfer Learning
- Reuse pretrained models (e.g., ImageNet) for new tasks.
- Approaches:
  1. **Fixed feature extractor** – freeze CNN layers, train new classifier.
  2. **Fine-tuning** – unfreeze and train entire or partial network.
  3. **Pretrained models** – use released checkpoints.

### Example (PyTorch)
```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
```

- **When to fine-tune**:
  - Small, similar dataset → freeze early layers.
  - Large, different dataset → train all layers.
  - Small, different dataset → use pretrained CNN as fixed extractor.

---

## References
- PyTorch Docs: https://pytorch.org/tutorials/
- Medium: CNNs architectures (LeNet–ResNet)
- Vaswani et al. (2021) – Vision Transformer
- Radford et al. (2021) – CLIP
