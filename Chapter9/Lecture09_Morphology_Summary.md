# Lecture 09: Morphological Image Processing
## Objectives
- Understand morphological image processing fundamentals.
- Apply techniques for segmentation, automated inspection, and image description.

---

## 1. Introduction
- Morphology analyzes and manipulate shapes/structures in images (binary) via mathematical operations.  
- Useful for segmentation, noise reduction, object extraction, and feature enhancement/removal.  
- Extracted features: boundaries, skeletons, convex hulls.  
- Used in pre/post-processing (filtering, thinning, pruning).

### Set Theory Basics
- **Union:** A ∪ B  
- **Intersection:** A ∩ B  
- **Complement:** Aᶜ  
- **Difference:** A − B = A ∩ Bᶜ  
- **Reflection:** Â = {w | w = −a, a ∈ A}  
    - สะท้อนไปอีกฝั่งผ่านแกน x = y
- **Translation:** A_z = {c | c = a + z, a ∈ A}
    - เลื่อน +z_x + z_y

---

## 2. Logic Operations with Binary Images
- Operate pixel-wise using logical AND, OR, NOT on binary matrices.
    - A or B = ทุก pixel ของทั้ง 2 รูปรวมกัน
    - A xor B = ส่วนที่ซ้ำกันเป็น 0
    - NOT A and B = ส่วนที่เป็น A เป็น 0 หมด
---

## 3. Dilation and Erosion

### Dilation (expands)  : พองตัว
- Adds pixels to object boundaries.  
- Formula: A ⨁ B = {z | (B̂)_z ∩ A ≠ ∅}  
- Structuring element acts like a convolution mask.  
- เอา mask ไปแปะถ้าใน mask มี 1 ตำแหน่งนั้นก็จะเป็น 1 สมมติ mask เป็น 3x3 (1 ล้วน) เท่ากับขยายทุก pixel ที่เป็น 1 ในรูป ให้ใหญ่ขึ้น 1 pixel, รูปทรง mask เท่ากับทำให้แต่ละ pixel ทีเป็น 1 กลายเป็นรูปทรงนั้น
- **Use:** Bridge small gaps and connect objects.

### Erosion (shrinks) : หด
- Removes pixels from object boundaries.  
- Formula: A ⊖ B = {z | B_z ⊆ A}  
- Dual of dilation.
- เอา mask ไปแปะถ้าใน mask มีตำแหน่งที่ตำแหน่งเป็น 0 จะเป็น 0 สมมติ mask เป็น 3x3 (1 ล้วน) เท่ากับหกทุก pixel ที่เป็น 1 ในรูป ให้เล็กลง 1 pixel, รูปทรง mask เท่ากับทำให้แต่ละ pixel ทีเป็นรูปทรงนั้นเป็น 1 pixel เดี่ยวๆ 
- **Use:** Remove small noise or irrelevant details.

```python
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
```

---

## 4. Opening and Closing

### Opening
- Smooths contours, breaks narrow connections, removes thin protrusions.  
- Erosion -> Dilation 
- Formula: A ∘ B = (A ⊖ B) ⨁ B

### Closing
- Fills small holes, closes narrow gaps, connects objects. 
- Dilation -> Erosion 
- Formula: A • B = (A ⨁ B) ⊖ B

**Duals:** (A ∘ B)ᶜ = Aᶜ • B̂

```python
cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

---

## 5. Hit-or-Miss Transformation : Shape Detection
- Detects specific shapes/patterns.  
- Formula: A ⊛ B = (A ⊖ B1) ∩ (Aᶜ ⊖ B2)  
- For finding object B1, B2 = dilated B1
- (A ⊖ B1) : เพื่อกำหนดอันเล็กกว่า
- (Aᶜ ⊖ B2) : เพื่อกำหนดอันใหญ่กว่า
- Used for locating defined shapes in binary images.
- Limitation : 
    - ถ้าภาพมีส่วนที่ซ้อนกันจะหาไม่ได้
    - ต้องรู้ขนาดและรูปทรงแน่นอนของรูปที่จะหา = ถึงสามารถทราบ location ได้
---

## 6. Basic Morphological Algorithms

### Boundary Extraction
B(A) = A − (A ⊖ B)
- (A ⊖ B) คือ A ที่เล็กลง เมื่อลบก็จะได้ขอบใน
- (A ⨁ B) - A : ได้ขอบนอก 
 
### Region Filling
Xₖ = (Xₖ₋₁ ⨁ B) ∩ Aᶜ : เติมสีในรูป หยุดเมื่อ ไม่มีอะไรเปลี่ยนแล้ว B ใช้เป็น +

### Connected Components
Xₖ = (Xₖ₋₁ ⨁ B) ∩ A : หยุดเมื่อไม่มีอะไรเปลี่ยน , หาก B เป็น 3x3 1 ล้วนแปลว่ามองแทยงก็เป็นชิ้นเดียวกันที่เชื่อมกัน

### Convex Hull
-   Smallest convex set containing A, using repeated hit-or-miss transforms.   
    - convex : ขีดเส้นจากจุดใดๆ ในรูปทรงไปอีกจุดในรูปต้องอยู่ในรูปทรง 100 %
    - มี 4 รูปทรงที่ต้องหา แล้วเอามารวมกัน 
---

## 7. Thinning, Thickening, Skeleton, Pruning

### Thinning
Removes outer layers to reduce objects to minimal connected lines.  
A⨂B = A − (A ⊛ B) แต่ B มี 8 ตัว

### Thickening
Dual of thinning.  
A⊙B = A ∪ (A ⊛ B) ใช้ B 8 ตัวเหมือน Thinning

### Skeleton
Set of points equidistant from boundaries. Represents shape with minimal information.

### Pruning
Removes small spurs left after thinning/skeletonization.
    - thinning ออกก่อนแล้วค่อยเอา endpoint มา dilation ก่อนจะ union กับผลของ thinning
---

## 8. Grayscale Morphology

### Operations
- **Dilation:** (f ⨁ b)(x,y) = max{f(x−s, y−t) + b(s,t)} : สว่างขึ้น
- **Erosion:** (f ⊖ b)(x,y) = min{f(x+s, y+t) − b(s,t)} : มืดลง
- **Opening:** f ∘ b = (f ⊖ b) ⨁ b  
- **Closing:** f • b = (f ⨁ b) ⊖ b

---

## 9. Applications
- **Morphological Smoothing:** Remove bright/dark noise (opening + closing).  
- **Morphological Gradient:** g = (f ⨁ b) − (f ⊖ b) : ขยาย - หด
- **Textural Segmentation:** Distinguish texture regions by structure size.  
- **Granulometry:** Estimate particle size distributions.

