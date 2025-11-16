# Lecture 12: Image Segmentation II  
**Instructor:** Punnarai Siricharoen, Ph.D.  

---

## 1. From Traditional to Modern
**Traditional methods:**  
- Thresholding  
- Region growing  
- Edge-based detection  
- Handcrafted features: HOG, SIFT + Machine Learning  

**Modern methods:**  
- Deep Learning (CNNs)

---

## 2. Modern Image Segmentation
**Types:**
- **Semantic Segmentation:** Assigns a class label to each pixel.  
- **Instance Segmentation:** Detects each object instance individually.  
- **Panoptic Segmentation:** Combines semantic and instance segmentation — produces pixel class + instance ID.

---

## 3. Datasets
| Dataset | Classes | Images | Description |
|----------|----------|---------|-------------|
| Cityscapes | 30 | 25K | Urban street scenes (fine + coarse annotations) |
| ADE20K | 150 | 20K | Scene-centric (sky, road, grass, person, car) |
| KITTI | 11 | 323 | Traffic scenarios, RGB + LIDAR |
| CamVid | 32 | 701 | Driving video frames |
| PASCAL VOC | 20 | 3K | 20 object categories |
| COCO-Stuff | 172 | 164K | Semantic segmentation + detection + captioning |

---

## 4. Evaluation Metrics
- **IoU (Intersection over Union):** Overlap between prediction and ground truth.  
- **Dice Score:** Similar to IoU but emphasizes overlap.  
- **Pixel Accuracy, Precision, Recall**  
- **mAP (Mean Average Precision):** Used in object/instance segmentation.

---

## 5. Deep Learning-based Segmentation
### 5.1 Fully Convolutional Networks (FCN)
- Replace dense layers with convolutional layers.  
- Use **transposed convolution** (upsampling/deconvolution).  
- Encoder–decoder architecture.

### 5.2 U-Net
- U-shaped encoder–decoder network with skip connections.  
- Encoder: halves spatial dimensions, doubles filters.  
- Decoder: doubles spatial dimensions, halves filters.  
- Uses **Dice loss** and **Binary Cross-Entropy with Logits loss**.  
- Can fine-tune pretrained backbones via `segmentation_models_pytorch`.

---

## 6. DeepLab Family
| Version | Key Feature |
|----------|-------------|
| **v1** | Atrous (dilated) convolution + CRF post-processing |
| **v2** | Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context |
| **v3** | Encoder–decoder structure, depth-wise separable convolutions |
| **v3+** | Enhanced decoder for boundary refinement, Aligned Xception backbone |

---

## 7. Depth-wise Separable Convolution
- Splits convolution into **depth-wise** and **point-wise (1×1)** operations.  
- Reduces parameters and computation.  
- Used in MobileNet and Xception architectures.

---

## 8. Advanced Architectures
### EfficientNet-NAS-FPN (2019)
- Neural Architecture Search for optimal FPN design.  
- Efficient backbone, multi-scale feature fusion, resource efficient.

### HRNet (2020)
- Maintains high-resolution features through parallel branches.  
- Multi-scale fusion, accurate and efficient.

---

## 9. State-of-the-Art Models (ADE20K 2023–2024)
- **ONE-PEACE:** Multi-modal general representation model (vision, audio, language).  
- **InternImage-H:** Large CNN with deformable convs, strong adaptability.  
- **M3I Pre-training:** Multi-modal pretraining via mutual information maximization.  
- **BEiT-3:** Multiway transformer for multimodal tasks.  
- **EVA:** Scaled visual representation learning.

---

## 10. Current Trends
- Applications: medical imaging, satellite vision, AR/VR, cartography.  
- Weakly supervised learning (image-level, bounding box, point annotations).  
- Domain adaptation for varying input types.  
- Multi-modal fusion (RGB + Depth/Thermal).  
- Real-time segmentation.

---

## Exercise
**Task:**  
Modify U-Net backbone → compare performance and justify choice.

---

**References:**  
- Sultana et al., *Knowledge-Based Systems*, 2020  
- Chen et al., *DeepLab Series*, 2017–2018  
- Minaee et al., *PAMI*, 2022  
- Wang et al., *HRNet, PAMI*, 2020  
- Mo et al., *Semantic Segmentation Review*, 2022  
