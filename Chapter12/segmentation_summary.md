
# Image Segmentation II 
---

## 1. Overview of Image Segmentation
### Traditional → Modern Evolution
- **Traditional**: Thresholding, Region Growing, Edge Detection  
- **Hand‑crafted ML**: HOG, SIFT + classifiers  
- **Deep Learning**: CNN-based segmentation dominates

---

## 2. Types of Segmentation
- **Semantic Segmentation**: Class label for every pixel  
- **Instance Segmentation**: Separate masks for each object instance  
- **Panoptic Segmentation**: Combines semantic + instance

---

## 3. Datasets
| Dataset | #Classes | Notes |
|--------|---------|-------|
| **Cityscapes** | 30 | Urban scenes, 25k images |
| **ADE20K** | 150 | Scene‑centric, widely used |
| **KITTI** | 11 | Autonomous driving |
| **CamVid** | 32 | Driving videos |
| **PASCAL VOC** | 20 | Classic benchmark |
| **COCO‑Stuff** | 172 | Scene understanding |

---

## 4. Evaluation Metrics
- **IoU (Intersection over Union)**  
- **Dice Score**  
- **Pixel Accuracy**  
- **Precision/Recall**  
- **mAP** (object/instance level)

---

## 5. Modern DL Approaches Timeline
Semantic / instance / panoptic segmentation with CNNs:
- FCN → U‑Net → DeepLab series → HRNet → FPN‑based methods → Large-scale foundation models

---

## 6. FCN (Fully Convolutional Network)
- Converts classification CNN → pixel classifier  
- Uses **transposed convolution (deconvolution)** for upsampling  
- No fully connected layers; keeps spatial structure

---

## 7. U‑Net Architecture
**Encoder–Bottleneck–Decoder**  
- Downsampling: Conv blocks + MaxPool  
- Upsampling: Transposed Conv  
- **Skip connections** keep high‑res spatial info  
- Popular for medical imaging and small datasets

### Training Losses
- **Dice Loss**
- **BCEWithLogitsLoss**
- Often combined for stability

---

## 8. Pretrained Models (segmentation_models_pytorch)
Common backbones:
- VGG16
- ResNet
- EfficientNet
- MobileNet  
Provides: U‑Net, FPN, LinkNet, PSPNet, DeepLabV3+

---

## 9. DeepLab Family
### *DeepLab v1*
- Atrous (dilated) convolution  
- CRF post‑processing  
- Keeps resolution without increasing parameters

### *DeepLab v2*
- **ASPP (Atrous Spatial Pyramid Pooling)**  
- Multi-scale context using parallel dilated conv

### *DeepLab v3*
- Encoder–decoder  
- Depthwise separable conv (like MobileNet/Xception)

### *DeepLab v3+*
- Strong decoder for sharper boundaries  
- Xception backbone

---

## 10. Depthwise Separable Convolution
- Factorizes standard conv:
  - **Depthwise Conv** (per channel)
  - **Pointwise Conv** (1×1)  
- Reduces params dramatically  
- Used in MobileNet, Xception, DeepLabv3+

---

## 11. EfficientNet‑NAS‑FPN
- **EfficientNet backbone** → computationally efficient  
- **NAS‑FPN** searches for best multi‑scale fusion  
- Strong for object detection + segmentation

---

## 12. HRNet
- Maintains **high‑resolution** feature maps throughout  
- Multi‑scale fusion  
- Excellent for fine detail boundaries

---

## 13. State-of-the-Art Models (ADE20K)
Recent SOTA trends:
- **ONE‑PEACE**
- **InternImage‑H**
- **BEiT‑3**
- **EVA**
- **M3I Pretraining**  
Large multimodal & unified vision models dominate.

---

## 14. Trends in Segmentation
- Real-time segmentation  
- Weakly supervised & semi-supervised data  
- Multi-modal fusion (RGB+T, RGB+D)  
- Domain adaptation  
- Large-scale pretrained multimodal models

---

