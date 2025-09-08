# Lecture 05: Image Restoration
---

## 📌 Objectives
1. เข้าใจประเภทต่างๆ ของ **image degradation**  
2. เข้าใจและเลือกใช้ **image restoration techniques** ให้เหมาะกับแต่ละ degradation  

---

## 📚 Core Concepts

### 1. Image Degradation Model
- ภาพที่เสื่อมสภาพเกิดจาก  
  - Noise (ค่าพิกเซลผิดเพี้ยน)  
  - Blur (out-of-focus, motion blur)  
- Model (spatial domain):  

$$ g(x,y) = h(x,y) * f(x,y) + \eta(x,y) $$  

- ใน frequency domain:  

$$ G(u,v) = H(u,v)F(u,v) + N(u,v) $$  

---

### 2. Noise Types
- **Salt & Pepper (Impulse noise)**  
  - จุดดำ-ขาวสุ่ม เหมือนเกลือพริกไทย  
  - PDF:  

  $$
  p(z) =
  \begin{cases}
  P_a & z = a \\
  P_b & z = b \\
  0   & \text{otherwise}
  \end{cases}
  $$  

- **Gaussian Noise**  
  - เกิดจาก random fluctuation ของสัญญาณ  
  - PDF:  

  $$
  p(z) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(z-\mu)^2 / (2\sigma^2)}
  $$  

- **Speckle Noise** (multiplicative)  
  $$ g(x,y) = f(x,y) \times s(x,y) $$  
  พบใน radar, ultrasound imaging  

- **Periodic Noise**  
  - จาก electrical / EM interference  
  - เห็นเป็น pattern ซ้ำๆ ใน spectrum  

---

### 3. Restoration Methods

#### (a) Spatial Filtering
- **Averaging / Low-pass filters** → ลด Salt & Pepper, Gaussian noise  
- **Median filter** → ดีมากกับ Salt & Pepper  
- **Gaussian filter** → blur เพื่อลด noise  

#### (b) Frequency Domain Filtering
- Band-reject filters → ลบ periodic noise  
- Notch filters → ลบ frequency เฉพาะจุด (ต้องทำเป็นคู่ symmetric)  

#### (c) Inverse Filtering
- Idea: หารด้วย H(u,v) เพื่อย้อน degradation  

$$ \hat{F}(u,v) = \frac{G(u,v)}{H(u,v)} $$  

- ปัญหา: ถ้ามี noise → amplify noise  

#### (d) Wiener Filtering
- Minimize Mean Square Error (MMSE):  

$$
\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K} G(u,v)
$$  

- K = noise-to-signal ratio constant  
- ใช้ได้ผลดีกว่า inverse filter  

#### (e) Super-resolution
- ใช้ deep learning / advanced methods เพื่อเพิ่ม resolution ของภาพ  

---

## 🐍 Python Examples

### Add Salt & Pepper Noise
```python
import cv2
from skimage.util import random_noise
import matplotlib.pyplot as plt

img = cv2.imread("twins.jpg", 0)

sp = random_noise(img, mode='s&p')
sp2 = random_noise(img, mode='s&p', amount=0.2)

plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(sp, cmap='gray'); plt.title("S&P noise")
plt.subplot(1,3,3); plt.imshow(sp2, cmap='gray'); plt.title("S&P noise 0.2")
plt.show()
```

### Median Filtering
```python
image_sp = (255*sp).astype('uint8')
restored = cv2.medianBlur(image_sp, 3)
plt.imshow(restored, cmap='gray')
plt.title("Median Filtered")
plt.show()
```

### Fourier Spectrum & Notch Filter
```python
import numpy as np

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude = np.log(1 + np.abs(fshift))

# Example notch filter (zeroing specific region)
f_notch = fshift.copy()
f_notch[90:110, 100:120] = 0
f_notch[140:160, 210:230] = 0

img_back = np.fft.ifft2(np.fft.ifftshift(f_notch))
plt.imshow(np.abs(img_back), cmap='gray')
plt.title("Notch Filter Result")
plt.show()
```

---

## ✅ Summary
- Image degradation → noise, blur, interference  
- Salt & Pepper, Gaussian, Speckle, Periodic noise → แต่ละแบบใช้ filter ต่างกัน  
- Spatial filters (average, median) = เบสิค restoration  
- Frequency domain filtering = powerful for periodic noise  
- Inverse filtering → เสี่ยง amplify noise  
- Wiener filtering → practical choice (balance noise + blur)  
- Super-resolution → วิธี modern ที่ใช้ deep learning  
