# Lecture 03: Image Histogram and Spatial Filtering
---

## 📌 Objectives
1. เข้าใจการอธิบายภาพด้วย **Histogram** และ **Histogram Equalization**  
2. เข้าใจเทคนิค **Filtering** เช่น image smoothing และ sharpening  

---

## 📚 Core Concepts

### 1. Histogram
- Histogram = กราฟแสดง distribution ของ intensity  
- ใช้ histogram manipulation เพื่อ  
  - **Enhancement** (contrast stretching)  
  - **Compression**  
  - **Segmentation**  
  - ให้ **statistics** เช่น min, max, mean intensity  

**Formula**:  
- Image f มี gray levels = 0,…,L-1  
- Histogram: `h(r_k) = n_k` (จำนวน pixel ที่ intensity = r_k)  
- Normalized Histogram: `p(r_k) = n_k / n`  
  - เป็น estimate ของ probability  
  - Σp(r_k) = 1  

---

### 2. Histogram Equalization
- Idea: ทำ histogram ให้ออกมา **flat** → กระจายค่า intensity ให้ contrast ดีขึ้น
- ใช้ **PDF** + **CDF** ในการหา mapping function `s = T(r)`  
- Formula:   $ s_k = (L-1) \sum_{j=0}^k p(r_j) $

- ทำให้ dark area → brighter, bright area → darker balance  
- ใช้เยอะใน medical image, satellite image  

---

### 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Equalization แบบแบ่งย่อยเป็น **grids** (ตัดเป็นส่วนๆ แล้วแยกทำแต่ละส่วน) 
- ใช้ clip limit ป้องกันไม่ให้ contrast extreme เกินไป  
- Steps:  
  1. แบ่งภาพเป็น grid  
  2. คำนวณ histogram ของแต่ละ grid  
  3. Clip ค่าเกิน limit  
  4. ใช้ CDF mapping ใหม่  
  5. Interpolation รวมกลับมาเป็นภาพเดียว  

→ ใช้ใน medical imaging, low-light enhancement  

---

### 4. Spatial Filtering : filter , mask, kernel, template or window (inside = coeff)
- **Neighborhood operation**: ใช้ kernel/mask คูณกับ pixel รอบๆ (convolution)
  - Image Border : Partial filter, Padding, Replicating, Perfectly filter  
- **Linear filtering** = convolution mask  
- **Non-linear filtering** เช่น median filter(reduce extreme noise impact) 

#### 4.1 Smoothing (Low-pass)
- Purpose: blur / ลด noise  
- Methods:  
  - Averaging filter (box filter)   -> blur edge, box filter คือ weighted ที่มีแต่ 1 
  - Weighted average (Gaussian mask)  -> อย่าลืม coeff รวมมาหารด้วย 
  - Median filter → ดีมากกับ impulse noise (salt & pepper)  
- ผลลัพธ์ขอบสีดำ ส่วนใหญ่เพราะ Padding ด้วย 0

#### 4.2 Sharpening (High-pass)
- Purpose: เน้นขอบ (edges), fine details  
- **Derivative based**: measure change in intensity
  - First-order (gradient → Sobel, Prewitt, Roberts) : measure rate of change of intensity
    - 1D : $\frac{df}{dx} = f(x+1) - f(x)$
    - Give edge direction + magnitude
  - Second-order (Laplacian)
    - 1D : $\frac{d^2f}{d^2x} = f(x+1) + f(x-1) - 2f(x)$ 

  - First-order derivative → gradient → highlights edges with direction.
  - Second-order derivative → Laplacian → highlights edges and points of rapid intensity change (but no direction = independent of the direction). 
- Idea: enhance discontinuities ของ gray level  

**Laplacian Mask Example**  
```
0  -1   0
-1  4  -1
0  -1   0
```

- Combine original + Laplacian → sharpen result  

**Result Mask is** เพราะต้องเอามารวมกับ original image
```
0  -1   0
-1  5  -1
0  -1   0
```

---

## 🐍 Python Examples

### Histogram Calculation
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("kitty.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
plt.bar(range(256), hist[:,0])
plt.show()
```

### Histogram Equalization
```python
equ = cv2.equalizeHist(img_gray)
plt.imshow(equ, cmap="gray")
plt.show()
```

### CLAHE
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img_gray)
plt.imshow(cl1, cmap="gray")
plt.show()
```

### Spatial Filtering (Averaging)
```python
import numpy as np

kernel = np.ones((3,3), np.float32) / 9
dst = cv2.filter2D(img, -1, kernel)
```

### Sobel Filter
```python
sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=3)
```

---

## ✅ Summary
- Histogram = เครื่องมืออธิบาย intensity distribution  
- Histogram Equalization = เทคนิคปรับ contrast (flat histogram)  
- CLAHE = adaptive equalization ป้องกัน over-enhancement  
- Spatial Filtering = ใช้ kernel กับ neighborhood →  
  - Smoothing (blur, reduce noise)  
  - Sharpening (edge detection, highlight detail)  
