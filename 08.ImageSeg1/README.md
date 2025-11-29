# Image Segmentation

## Objectives
- Understand segmentation fundamentals and challenges  
- Apply segmentation techniques to different problems  
- Identify limitations of each technique  

---

## 1. What is Image Segmentation
- Segmentation = Extract something from image
- Divides an image into meaningful regions or objects (finding group of pixel)
- Accuracy affects later analysis 
    - counting : number of each type of object
    - measuring : area, perimeter of objecs in the images
    - property study : intensity, texture
- Types:  
  - **Semantic segmentation:** classificatioin  
  - **Instance segmentation:** separate object instances -> Counting   
  - **Panoptic segmentation:** combine both  

---

## 2. Segmentation Properties
2 Main properties of intensity values : 
- **Discontinuity:** based on abrupt intensity change (edges) 
  - Edge-based : Detecting edges that separate region from each other
- **Similarity:** group similar pixels (intensity, color, texture) 
  - Thresholding : Using pixel intensity 
  - Region-based : Grouping similar pixels -> region growing/merge&split

---

## 3. Discontinuity-Based Methods

### Point & Line Detection
- Detect isolated points or lines using masks/kernel
- Masks detect horizontal, vertical, or diagonal lines : 
  Point : Laplacian mask 
  Line : Line mask
- **FYI** : Spatial and Convo filtering will give same result if its symmetric

### Edge Detection
- Find boundaries between regions  
- **First Derivative (Gradient):** Sobel / Prewitt  
  - Prewitt is simpler, but Sobel have noise suppression
- **Second Derivative:** Laplacian or LoG (zero crossings)  
  - LoG -> Thresholding -> Zero Crossing
- **Canny Edge Detector (1986):** can detect thin line
  1. Gaussian smoothing : noise reduction
  2. Compute gradient : Sobel in both hori&verti direction
  3. Non-maximum suppression : make edge thinner
  4. Hysteresis thresholding → connect real edges  
      - Only sure edge will be used : Double Hysteresis
      - Edge tracking : only edge that connect with sure edge will remain

---

## 4. Hough Transform
Detect geometric shapes (curve line) from edge points.
  - Find every possible line that fit a set of pixels in an image.
  - each pixel can have 180 line create hough line, combined hough line of entire picture will show position of point that intersect most -> then we got a line 

### Line Detection
- Line equation: r = xcosθ + ysinθ  
- Uses accumulator to detect peaks  

### Circle Detection
- Parameters: x₀, y₀, r  
- Used also for ellipse, parabola detection  

**Applications:** lane detection, shape recognition  

---

## 5. Similarity-Based Methods

### Thresholding
Separate object and background by intensity.
- The problem is how to choose threshold.
    - Too low -> reduce object size
    - Too high -> include extraneous background
  So we need automatic method for choosing -> Otsu's Method

#### Types
- **Global threshold:** one Threshold for all pixels  
- **Local threshold:** divide image into non-overlapping sections, each section got fix threshold  
- **Adaptive threshold:** For each pixel, the threshold is computed from its local neighborhood (sliding window).  

#### Otsu’s Method
- Automatic threshold from histogram (probability distribution)  
- Step through every possible K, each step calculate variance of 2 groups (group#1 = 0 to k, group#2 = k+1 to max-intensity)
- Maximizes between-class variance  

---

## 6. Region-Based Segmentation

### Region Growing / Watershed
- Pixels grouped based on similarity  
- Watershed treats image as landscape:  
  - Catchment basins = valleys, region  
  - Watershed lines = boundaries, edge  
- Marker-controlled version prevents over-segmentation  

**Applications:** roads, object separation, fracture detection  

### Region Splitting & Merging
- **Splitting:** divide large region until homogeneous  
- **Merging:** combine adjacent similar regions  
- Example rule: |max - min| ≤ T  

