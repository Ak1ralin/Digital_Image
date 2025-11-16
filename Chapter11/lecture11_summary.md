# Lecture 11: Object Recognition II

## Objectives

-   Learn how to improve models using **augmentation techniques**
-   Understand the **evolution of deep learning models**
-   Learn how to reuse models via **transfer learning**

## Visualizing Convolutional Networks

### Calculation 
- Conv2d : params = (input_channel\*output_channel\*k_h\*k_w)+output_channel
    - Every input channel will need #output_channel kernels -> (input_channel\*output_channel\*k_h\*k_w)
    - Each output_channel have 1 bias -> + output_channel
    - `output_channel = sum_over_channels(convolution_results) + bias_for_this_filter`
    
### Filters

-   CNN filters uncover what each layer learns.
-   Early layers detect edges/textures; deeper layers detect object
    parts.
-   Kernel size affects spatial sensitivity.

### Feature Maps

-   Show regions the model attends to.

## Image Transformation & Augmentation

Why augment? - Expand dataset size artificially. - Reduce overfitting. -
Improve model generalization.

Common augmentations: - Flips, rotations, scaling, cropping,
translation, noise.

PyTorch example:

``` python
transforms.Compose([
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation((-30, 30)),
    transforms.ToTensor(),
])
```

## Evolution of Deep Learning

### LeNet-5 (1998)

-   Early CNN for digits.

### ImageNet & ILSVRC

-   14M+ images; benchmark challenge.

### AlexNet (2012)

-   Top-5 error: 15.3%
-   ReLU, dropout, data augmentation.

### GoogLeNet/Inception (2014)

-   Top-5 error: 6.67%
-   Inception modules : 
    - Concat result of different feature extract method
        - 1x1 Convo 
        - 1x1 Convo -> 3x3 Convo -> use 1x1 first to reduce number of params
        - 1x1 Convo -> 5x5 Convo -> use 1x1 first to reduce number of params
        - Maxpooling -> 1x1 Convo -> to ensure same output_channel with others
-   1×1 conv (Depth-wise Convo)-> reduce number of channel
    - 256 input_channel with 3x3 kernel ConVo = 256x3x3 + 1(bias) = 2305 params per filter (output_channel) = 2305 x 32 = 73760
    - use 1x1 conv we can reduce number of input_channel 
        - 256 channels to 32 channels = 256 x (32 x 1 x 1 + 1) = 8448 params    
-   Auxially Output : Use in training phase to check itermediated result in before final result.

### VGGNet (2014)

-   Just Bigger Model. Nothing New.
-   138M+ parameters.

### ResNet (2015)
- ResNet ถูกออกแบบมาเพื่อแก้ปัญหา **deep network train ยากเพราะ gradient vanishing**
- Sol : เพิ่ม **skip connection (shortcut)** ให้ gradient ไหลตรงกลับได้ และให้ block เรียนรู้แค่ “สิ่งที่ต้องแก้ (Residual)”
    - Output = **F(x) + shortcut(x)** → ทำให้ backprop เรียนเฉพาะส่วน F(x) ที่ต้องปรับ
    - shortcut(x) มาจาก skip connection
        - **Identity**: ใช้เมื่อ spatial size และจำนวน channel เท่ากัน → บวกตรงได้
        - **1×1 Conv Shortcut**: ใช้เมื่อขนาดหรือ channel ไม่เท่ากัน → ปรับด้วย `Conv1×1` (และ BN) + `stride` ให้ขนาด match ก่อนบวก
- BottleNeck Block แทนที่จะใช้ Convo แปลงตรงๆ , ใช้ 1x1 Convo ครอบก่อนจะทำให้ใช้ params ลดลงมาก, ได้ depth กว่ามากเมื่อใช้ params จำนวนใกล้กัน

- ResNet50 much better than ResNet34, while params is closer -> because ResNet50 have BottleNeck Block

### ResNeXt (2017)

- **ResNeXt = ResNet + Grouped Convolution (เพิ่ม cardinality (กี่ paths)) : ไม่ได้ใช้  Convo Block ปกติ** 
    - แนวคิดหลัก: “split–transform–merge”
    - ใช้โครงสร้าง Residual Block แบบ ResNet แต่ **แทนที่ 3×3 Conv ปกติด้วย 3×3 Grouped Conv**
- Normal Conv vs Grouped Conv 
    - **Normal Convolution**
    - ทุก output channel ได้ข้อมูลจาก **ทุก input channel**
    - เช่น input 3 ch (ic1,ic2,ic3) → output 3 ch (oc1,oc2,oc3)  
        - ต้องมี kernel = 3×3 = **9 kernels** (k1 - k9)
        - `oc1 = Relu(BN(ic1Convk1 + ic2Convk2 + ic3Convk3))` -> **oc1 ได้ข้อมูลจากทั้ง 3 channel**
    - ข้อเสีย: params เยอะ, representation ผสมกันหมด

    - **Grouped Convolution**
    - แบ่ง channels เป็นหลายกลุ่ม แล้วคำนวณแยกกัน
        - group1: `oc1 = ReLU(BN( ic1 * k1 + ic2 * k2))`
        - group2: `oc3 = ReLU(BN( ic3 * k3 + ic4 * k4))`
        - → **oc1 ไม่ได้ข้อมูลจากทุก channel แต่เฉพาะกลุ่มของมัน**
    - ข้อดี: params ลดลง, ได้หลาย “แบบของ feature” พร้อมกัน  
        (หัวใจของ ResNeXt คือเพิ่ม cardinality = จำนวน group)

- Group Convolution ดียังไง
    - Conv ปกติ: ทุก output channel ต้องคุยกับทุก input channel → params เยอะมาก
    - Grouped Conv: แบ่ง input channels เป็นหลายกลุ่ม → ทุก group ประมวลผลแยกกัน → แล้วค่อย merge
        - **ลดพารามิเตอร์ลงแรง**
        - **เพิ่มความหลากหลายของ representation** (แต่ละ group เรียนรู้อะไรคนละแบบ)
        - ได้ performance ดีขึ้นโดยไม่ต้องเพิ่ม width/depth
- Result
    - accuracy ดีขึ้นมากเมื่อเทียบกับ ResNet ที่ใช้ params ใกล้กัน
    - scale ได้ดี (เพิ่ม cardinality ไม่บานปลายเหมือนเพิ่ม width หรือ depth)
    - ใช้โครงสร้างคล้าย ResNet → train ง่าย, optimization เหมือนเดิม

### EfficientNet (2020)

-   EfficientNet = (MBConv + SE) + Compound Scaling (สูตรคณิตศาสตร์สำหรับขยายโมเดลให้โตแบบคุ้มสุด)
    - Compound Scaling
        - คือ “สูตรขยายโมเดล” (depth/width/resolution) แบบ balance
            - มี 4 ตัวแปร:
                - φ = factor ที่บอกว่าอยากขยายแค่ไหน
                - α = น้ำหนักของ depth 
                - β = น้ำหนักของ width
                - γ = น้ำหนักของ resolution
            - แล้วใช้สูตร:
                - depth = α^φ
                - width = β^φ
                - resolution = γ^φ
        - ไป grid search หา α, β, γ ที่ดีที่สุดบน ImageNet 
        - B0 -> B7 : เพิ่มค่า φ (scaling factor) ทีละขั้น
    - MBConv (Mobile Inverted Bottleneck Convolution)
        - มันคือ block ที่มาจาก MobileNetV2
        - จุดเด่นคือ โคตรประหยัดพลัง แต่เก็บพลังการเรียนรู้ไว้ครบ
    - SE (Squeeze-and-Excitation)   
        - เหมือนใส่ soft attention ให้ block&network บอกได้ว่า “ช่องไหนสำคัญกว่า” → accuracy พุ่งแบบแทบไม่เพิ่ม compute
            - ให้ model โฟกัสเฉพาะ channel ที่สำคัญ
            - ลด noise จาก channel ที่ไม่ relevant
            - ทำให้ block ฉลาดขึ้นแบบแทบไม่เพิ่ม cost

### Vision Transformer (ViT, 2021)

-   Slide image to **Patch** (token of image) -> flatten
-   Use linear projection (NN) to any size (#features)
-   Positional Embedding : Sin/Cos, or learnable params
-   Add class token 
-   Layer Norm : 2 params per feature
-   MultiHeadAttention : separate feature group
-   MLP head : ใช้ class token -> classifier (NN)

## CLIP

-   Contrastive training with image--text pairs.
-   Zero-shot classification.
-   Joint image encoder + text encoder.
    - Training 
        - We have image--text pairs so diagonal should be highest
        - loss-function & backprop -> should make diagonal be highest 
        - make image and text encoder that will give similar vector for similar pair.

## Transfer Learning

### 1. Fixed Feature Extractor - Only Train Classifier

-   Small Dataset should use.
-   Freeze CNN layers (Use Pretrained Encoder); train classifier on top. 

### 2. Fine-tuning - Finetuning Selected Layer

-   Train selected layers. 
-   Strategy depends on dataset size & similarity.

### 3. Pretrained Models

-   Large Dataset and Different from original Dataset
-   Fully Fine-tuning , Initialized Weight from pretrained

