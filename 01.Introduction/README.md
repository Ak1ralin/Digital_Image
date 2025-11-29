# Introduction to Digital Imaging & Image Processing

- **Outline**
  - Introduction to digital imaging, applications, and challenges 
  - Principal imaging modalities (EM spectrum, sound, computational)

---

## Core Concepts

### What is Digital Imaging?
- Converting real-world signals (light, X-rays, sound, etc.) into digital images.
- Represented as a 2D function `f(x, y)` where amplitude = pixel intensity.

### Image Processing Pipeline
1. **Acquisition** – capture images via sensors.
2. **Storage & Compression** – efficient saving.
3. **Display/Printing** – visualization.
4. **Improvement** – enhancement, restoration, superresolution.
5. **Understanding** – segmentation, feature extraction, object detection, recognition.
6. **Generation** – creating synthetic or AI-generated images.

### Evolution of Image Processing
- **1980s** – Rule-based (edges, thresholds). ทำได้แค่งานง่ายๆ
- **2000s** – Feature-based (HOG, SIFT, Haar). feature บอกลักษณะพิเศษ แต่คนคิด feature นะ
- **2012+** – Deep learning (CNNs, ResNet, YOLO). CNNs เรียนรู้หา feature จากข้อมูล
- **2020+** – Multimodal AI (Vision + Language → CLIP, BLIP, GPT-4o). 
- **2023+** – Vision + Generation + Action (DALL·E, SAM, Robotics with vision).

---

## Challenges in Computer Vision
- Point of view, illumination (light change color), scale (size) ทำให้สิ่งเดียวกันดูต่างในมุมของรูป
- Deformation, occlusion (blocking), background clutter 
- Ambiguity (blur), motion, intra-class variations (รูปทรงต่างแต่เป็นเก้าอี้เหมือนกัน)

These make tasks like object recognition or segmentation difficult compared to human perception.

---

## Applications of Digital Imaging

### Daily Life
- Face detection & biometrics
- Hands-free selfies
- OCR (Optical Character Recognition)
- Object classification on mobile phones

### Advanced Applications
- 3D reconstruction ("Building Rome in a Day")
- Kinect sensor (depth, RGB, IMU)
- AR/VR, robotics, autonomous cars
- Medical imaging (CT, MRI, PET)
- Astronomy and satellite imaging (e.g., LANDSAT, weather monitoring)
- Industrial inspection

---

## Imaging Modalities (by Spectrum/Source) แต่ละคลื่นทำอะไรบ้าง

- **Gamma-ray Imaging** – PET, nuclear medicine, astronomy. แผ่รังสีแล้วตรวจจับ
- **X-ray Imaging** – CAT scans, angiography, industrial inspection. ดูการดูดซึม 
- **Ultraviolet Imaging** – microscopy, astronomy, laser studies.
- **Visible & Infrared** – microscopy, remote sensing, Weather observation.
- **Microwave** – Radar, Penetration through surfaces.
- **Radio Band** – MRI (Magnetic Resonance Imaging) Medical.
- **Other Modalities** – ultrasound, electron microscopy, computer-generated images.

## Using Sound 
- Low freq than wave, more safe than electromagnetic wave
---

## Python Exercises

### Libraries
- `scikit-image` – image processing algorithms.
- `NumPy` – array manipulation.
- `Matplotlib` – visualization.
- `OpenCV` (`cv2`) – computer vision.
- `pydicom` – medical DICOM images. medical image standard
```python
from skimage import io,color
import matplotlib.pyplot as plt
```

### Examples
1. **Read & Show Image**
   ```python
   img = io.imread("kitty.jpg") # make it be numpy (h,w,channels) 
   io.imshow(img) 
   io.show()
   plt.imshow(img[x:x+dx,y:y+dy]) # crop
   plt.show()
   ```

2. **Convert to Grayscale**
   ```python
   img = io.imread("kitty.png")
   gray = color.rgb2gray(img)

   plt.subplot(1,2,1); plt.imshow(img)
   plt.subplot(1,2,2); plt.imshow(gray, cmap="gray",vmin=0,vmax=255) 
   plt.show()
   ```

3. **OpenCV**
   ```python
   img = cv2.imread("kitty.jpg")
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   plt.imshow(img_rgb)
   plt.show()
   ```

4. **DICOM Medical Image**
   ```python
   from pydicom import dcmread
   from pydicom.data import get_testdata_file

   fpath = get_testdata_file('CT_small.dcm')
   ds = dcmread(fpath)
   print(f"Patient ID: {ds.PatientID}, Modality: {ds.Modality}")
   plt.imshow(ds.pixel_array, cmap="gray")
   plt.show()
   ```


