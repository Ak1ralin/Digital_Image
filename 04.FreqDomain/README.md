# Frequency Domain
---

## Objectives
1. เข้าใจ **frequency domain** สำหรับการวิเคราะห์ภาพ  
2. ใช้ **Fourier Transform** อธิบาย frequency domain ของภาพ  
3. ประยุกต์ **basic filters** บนภาพใน frequency domain  

---

## Core Concepts

### 1. Fourier Transform Basics
- ไอเดียหลักจาก **Jean-Baptiste Fourier (1807)**  
  → Any function that periodically repeats itself can be expressed as the sum of sines and/or consines of different freq.   
- ใช้เป็น **Mathematical Prism**: Decompose a signal into various frequencies.  
- Signal ที่ไม่ periodic แต่ finite energy → แทนได้ด้วย integral ของ sine/cosine  and weighted functions. 

**Euler’s Formula**  
$ e^{j\theta} = \cos\theta + j\sin\theta $

---

### 2. Discrete Fourier Transform (DFT)
- Digital image → discrete function ของ 2 variables (x,y)  
- **1D DFT**:  

$ F(u) = \frac{1}{M}\sum_{x=0}^{M-1} f(x) e^{-j2\pi ux/M} $  

- **2D DFT** สำหรับภาพ MxN:  

$ F(u,v) = \frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) e^{-j2\pi (ux/M + vy/N)}$  

- Inverse DFT: เอากลับมาภาพ spatial domain  
- Fourier Spectrum = magnitude  
- Phase Spectrum = phase angle  

- DC Component (u,v=0,0) = ค่าเฉลี่ย intensity  

---

### 3. Properties of Fourier Transform
- Real image → FT conjugate symmetric F(u,v) = F*(-u,-v)
- Shifting → คูณด้วย (-1)^(x+y) กับ ภาพก่อนค่อย FT ได้แบบ shift center เลย 
- Convolution theorem: convolution ใน spatial domain ↔ multiplication ใน frequency domain  

---

### 4. Filtering in Frequency Domain ： ทำเพราะมัน easier + cheaper ใน Spatial Domain 
- Workflow:  
  1. Multiply image by (-1)^(x+y)  
  2. คำนวณ DFT → F(u,v)  
  3. Multiply ด้วย filter H(u,v)  
  4. iDFT  
  5. Multiply (-1)^(x+y) อีกครั้ง  

#### Low-Pass Filter (LPF)
- ตัด high-frequency → blur, smooth  
- Types:  
  - **Ideal LPF**: binary mask (cutoff radius D0)   D0 น้อย = blur
  - **Butterworth LPF**: smooth transition, order n  n ยิ่งมากยิ่งชัน
  - **Gaussian LPF**: smoothest, no ringing  D0 น้อย = blur 

#### High-Pass Filter (HPF)
- ตัด low-frequency → sharpen edges  
- Ideal / Butterworth / Gaussian HPF = 1 - LPF, 3 ประเภทเหมือน LPF

#### FT Application
- Image Enhancement
- Image Compression : Only Low-freq cuz most info in Low-freq
- Machine Learning 
---

### 5. Spatial vs Frequency Filtering
- Spatial domain → convolution (kernel sliding)  
- Frequency domain → multiplication (FT(image) × FT(filter))  
- ผลลัพธ์ใกล้กัน แต่ frequency domain มักเร็วกว่าเมื่อภาพใหญ่ (FFT ช่วย)  

---

## Python Examples

### Compute Fourier Transform
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cameraman.tif', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(1+np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()
```

### Ideal Low-Pass Filter
```python
nx, ny = (256, 256)
x = np.linspace(-nx/2, nx/2-1, nx)
y = np.linspace(-ny/2, ny/2-1, ny)
i, j = np.meshgrid(x, y, sparse=True)

D0 = 15  # cutoff frequency
ideal_filter = np.sqrt(i*i + j*j) < D0

fshift_filtered = fshift * ideal_filter
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
img_back = np.abs(img_back)
plt.imshow(img_back, cmap='gray')
plt.title('Ideal LPF Result')
plt.show()
```

### Butterworth Low-Pass Filter
```python
D0, n = 15, 2
btw_filter = 1 / (1 + ((np.sqrt(i*i+j*j)/D0)**(2*n)))

fshift_filtered = fshift * btw_filter
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
plt.imshow(np.abs(img_back), cmap='gray')
plt.title('Butterworth LPF Result')
plt.show()
```

### Gaussian High-Pass Filter
```python
D0 = 15
gaussian_lpf = np.exp(-(i*i+j*j)/(2*(D0**2)))
gaussian_hpf = 1 - gaussian_lpf

fshift_filtered = fshift * gaussian_hpf
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
plt.imshow(np.abs(img_back), cmap='gray')
plt.title('Gaussian HPF Result')
plt.show()
```

---
