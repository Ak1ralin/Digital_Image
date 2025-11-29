# Wavelets and Multiresolution Processing
---

## Objectives
1. เข้าใจพื้นฐานของ **wavelets** (โดยเฉพาะ Haar wavelet)  
2. ประยุกต์ **wavelet transform** สำหรับ multiresolution analysis  

---

## Core Concepts

### 1. Introduction
- Fourier Transform → ให้ข้อมูลเชิง global ทุก frequency  
- แต่ **wavelets** → สามารถให้ทั้ง **spatial + frequency** พร้อมกัน  
- ใช้แก้ปัญหา object เล็ก/contrast ต่ำ และ object ใหญ่/contrast สูง  
- Motivation: วิเคราะห์ภาพในหลาย resolution  

---

### 2. Image Pyramids
- แสดงภาพในหลาย resolution  
- Base (high res) → Apex (low res) 
- Base level = $J = \log_2N$ 
- Pyramid level j: size = 2^j x 2^j (0 ≤ j ≤ J)  
- Gaussian pyramid ใช้ low-pass filter ก่อน subsample → ป้องกัน aliasing  
- Laplacian pyramid = residuals ระหว่าง Gaussian levels  

---

### 3. Basis Functions
- Wavelet basis สร้างจาก **mother wavelet** (เช่น Haar, Mexican Hat)  
- Orthonormal wavelets → ใช้เป็น basis decomposition  
- Haar wavelet: ใช้ average + difference (horizontal, vertical, diagonal)  

---

### 4. Fast Wavelet Transform (FWT)
- Efficient implementation ของ **Discrete Wavelet Transform (DWT)**  
- **1D FWT** → decomposition เป็น approximation + detail coefficients  
- **2D FWT** → ใช้กับ image: decomposition → (LL, LH, HL, HH)  
  - LL = Approximation  
  - LH = Horizontal detail  
  - HL = Vertical detail  
  - HH = Diagonal detail  

---

### 5. Applications of Wavelets
- Feature extraction → palmprint recognition  
- Image denoising, enhancement, compression  
- Extract structure vs noise → ใช้ GAN สร้างภาพคุณภาพสูง  
- ใช้ใน Neural Network: wavelet kernel / activation  

- Art analysis → ใช้ wavelet decomposition วิเคราะห์ศิลปะ  
- AI → crop disease / wheat disease classification  

---

## Summary
- Wavelets = Multiresolution representation (spatial + frequency info)  
- Image pyramid (Gaussian, Laplacian) → ตัวอย่าง multiresolution  
- Haar wavelet = simple case (average + difference)  
- FWT = efficient DWT → ใช้ได้ทั้ง 1D, 2D  
- Applications → feature extraction, denoising, compression, AI  
---

## Python Examples

### 2D Haar Wavelet Decomposition
```python
import pywt
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("kitty.png", 0)

LL, (LH, HL, HH) = pywt.dwt2(img, "haar")
titles = ["Approximation", "Horizontal", "Vertical", "Diagonal"]

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i+1)
    ax.imshow(a, cmap="gray", interpolation="nearest")
    ax.set_title(titles[i])
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
```

### Investigate Coefficient Stats
```python
print("LL min/max:", LL.min(), LL.max())
print("LH min/max:", LH.min(), LH.max())
print("HL min/max:", HL.min(), HL.max())
print("HH min/max:", HH.min(), HH.max())
```

### Segmentation Task
- Apply wavelet transform → ใช้ threshold บน high-frequency subbands (LH, HL, HH)  
- แยก object (e.g. cat) ออกจาก background  

---
