# Image Formation, Colors, and Arithmetic Operations
---

## Objectives
1. Learn how an image is formed.
2. Understand different color models and their purposes.
3. Apply arithmetic operations to process an image.

---

## Core Concepts

### 1. Image Formation
- **Human Eye**: 
  - Lens focuses light onto the retina. retina is where image actually form.
  - Retina have 2 types of receptors
    - Cones (6-7M) → color + details (at the central of retina).  
    - Rods (75-150M) → overall, sensitive to illumination, no color perception distribute all over the retina.

    BTW, image on retina is up-side down, brain flip it
- **Electromagnetic Spectrum**: visible light (400-700nm) is just one portion.
- **Image Acquisition Devices**: 
  - Single sensor (requires motion), line sensor, array sensor (e.g., CCD cameras).  
    - sensor is made from filter + sensing material (CMOS,CCD) + power
  - Image = combination of illumination + reflectance. ภาพเกิดจากแสงสะท้อนของวัคถุ

### 2. Sampling & Quantization : From continuous sensed data to digital form
- **Sampling** → digitizing coordinates. **position**
  - จากภาพที่มีลายละเอียดยิบย่อย แปะ grid แล้วเอาค่าเฉพาะตำแหน่งที่ตรง grid , Resolution
- **Quantization** → digitizing amplitude.  **value**
  - map จากค่าที่ละเอียดนับไม่ถ้วนเป็นสีต่างๆ , Bit-depth
- Digital image: `f(x,y)` with M rows, N columns, pixel intensities `[0 … L-1]`.  
- **Resolutions**:  
  - Spatial resolution → smallest discernable detail.  
  - Gray-level resolution → smallest intensity difference (e.g., 8-bit = 256 levels). L = 256

---

### 3. Color Fundamentals
- Color = powerful attribute for recognition.  
- **Achromatic light** → intensity. 
- **Chromatic light (400–700nm)**:  
  - Radiance = source energy (sun, light)
  - Luminance = observed energy (observed energy got from obj) 
  - Brightness = perceived color sensation (color sensation in retina) 
- **Human Vision**:  
  - Cones: ~65% red, 33% green, 2% blue. (but blue most sensitive) 
  - Colors form from variable RGB (Primary color) combinations. 
  - Secondary Color : Magenta, Cyan, Yellow 
- Attributes: 
  - **Brightness (Intensity)** : Achromatic component 
  - **Hue (Dominant Wavelength)** : Chromatic component
  - **Saturation (Amount of white light)** : Chromatic component
  - Hue and Saturation : **Chromaticity** 

#### How to represent color
- **CIE Chromaticity Diagram** → standard color representation.
  - z = 1 - (x + y) , x = red , y = green
- **Tristimulus** : Amount of red, green and blue to form color
  - $x = \frac{X}{X+Y+Z}$ , $y = \frac{Y}{X+Y+Z}$ , $z = \frac{Z}{X+Y+Z}$
  - $x + y + z = 1$
---

### 4. Color Models
- **RGB**: additive, displays/screens.  
  - All color normalised to [0,1], 0 is black 1 is white
  - Additive = light mixing → white when maxed
- **CMY/CMYK**: subtractive, printing. Secondary Color of light
  - Subtractive = pigment/ink mixing → black when maxed
- **HSI/HSV**: separates hue, saturation, intensity – aligns with human perception. close to how human interpret colors, ideal tool for image processing algorithm.
  - Hue : pure color
  - Saturation : degree hue diluted by white light
  - Brightness : Intensity
- **Gray** = 0.3R + 0.6G + 0.11B
- **L\*a\*b\***/CIELAB model: device-independent, L* = lightness, a* = green-red, b* = blue-yellow.

---

### 5. Arithmetic Operations in Spatial Domain
- Operates directly on pixels: 1. piece-wise 2.neighboring
  
  `g(x,y) = T[f(x,y)]` = `s = T(r)`
- **Point Operators**: multiplication (gain), addition (bias).  
  - Gain controls contrast. เพราะทำให้ผลต่างระหว่าง pixel ที่ต่างเพิ่มขึ้น
  - Bias controls brightness. เพราะทำให้ค่าเพิ่มขึ้นหมด

  `g(x,y) = af(x,y)+ b`
- **Gray-Level Transformations**: Image Enhancement
  1. Negative transformation `s = (L-1) - r` 0 -> 255 , 255 -> 0
  2. Log transformation (expand dark regions) `s = clog(1+r)`, r = normalised intensity
      - log0 ไม่มีค่า so + 1
      - log ทำให้ค่าลดลง -> brighter
  3. Power-law (gamma correction)  $s = cr^Y$ 
      - c, Y = positive constant
      - Y > 1 Brighter, Y < 1 Darker
      - ใช้กับ MRI Y = 0.4,0.6 more detailed becomes visible 
  4. Piecewise-linear (contrast stretching) ไม่ได้ใช้ function เดียวสำหรับทุก intensity แต่มีหลายฟังก์ชั่นแยกเป็น range
  5. Gray-level slicing : เอาเฉพาะ intensity บางส่วนที่ interest ส่วนอื่นให้มันหลายเป็น 0 หรือคงค่าเดิมไรงี้ เป็น Piecewise-linear แบบหนึ่ง 
  6. Bit-plane slicing : แยกตาม bit เอาเฉพาะบาง bit (MSB) -> image compression 

- **Logic/Arithmetic Ops**:  
  - Subtraction → background subtraction. g(x,y) = f(x,y)foreground - h(x,y)background
  - Averaging → noise reduction (astronomy). เอาหลายๆ รูปมา avg เพื่อลด noise 
  - AND/OR masking → select ROIs (Region of interest).

- **Interpolation**:
  - Nearest Neighboring : Just pick a nearest one, so blocky.
  - Linear : Averaging so smooth but blurry transition
  - Bicubic : Averaging but also fit with polynomial curve, so might have exceed.
  - Area : Nearest + Averaging
---

## Python Exercises

### Quantization
```python
from skimage import io, color
import numpy as np

img = io.imread("kitty.jpg")
gray = color.rgb2gray(img)
gray255 = (gray * 255).astype(int)

gray255[gray255<64] = 0
gray255[(gray255>=64) & (gray255<128)] = 1
gray255[(gray255>=128) & (gray255<192)] = 2
gray255[gray255>=192] = 3
```

### Shrink & Zoom with OpenCV
```python
import cv2

SIZE = 64
img = cv2.imread("kitty.jpg")

# Shrink to 64x64 (nearest & area)
img_nn = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
img_area = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

# Zoom to 1000x1000 (bicubic & nearest)
img_bicubic = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_CUBIC)
img_nn_zoom = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_NEAREST)
```

### Arithmetic Operations (Gain & Bias)
```python
import cv2
import numpy as np

GAIN = 2
BIAS = 50

img = cv2.imread("kitty.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gain
img_mul = cv2.multiply(img_gray, GAIN)
# Bias
img_add = cv2.add(img_gray, BIAS)
```

### Color Model Conversion
```python
# RGB -> HSV, YCbCr, Lab
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
```

