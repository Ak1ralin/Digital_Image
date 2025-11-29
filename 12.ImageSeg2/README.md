# Image Segmentation II

## Big Picture

- ต่อจาก Lecture 8 แต่โฟกัสที่ **modern image segmentation ด้วย Deep Learning**
- เข้าใจ
    - ประเภท segmentation (semantic / instance / panoptic)
    - metric วัดผล (IoU, Dice, Pixel Acc, Precision/Recall, F1)
    - สถาปัตยกรรมหลัก: **FCN, U-Net, DeepLab family**
    - ไอเดียสำคัญอย่าง **transposed conv, atrous/dilated conv, depth-wise separable conv**
    - แนวโน้ม model ปัจจุบัน (ONE-PEACE, InternImage, BEiT-3, EVA ฯลฯ)

---

## Types of Modern Segmentation

- **Semantic segmentation : นับไม่ได้**
    - ให้ label กับ “ทุก pixel” → pixel-level classification
    - pixel ที่เป็น class เดียวกันถือว่าเหมือนกัน ไม่แยก instance
- **Instance segmentation : นับได้**
    - แยก “แต่ละ object” ออกจากกัน (คนสองคนสอง mask)
- **Panoptic segmentation**
    - รวมสองอันข้างบน
    - output มีทั้ง
        - class ของ pixel (semantic)
        - id ของ instance (instance)

---

## Metrics สำหรับ Segmentation

### IoU (Intersection over Union)

- ใช้วัด overlap ของ prediction กับ ground-truth, Strict ค่ามักต่ำ แต่ค่อนข้างเป็น Standard
- สูตร:  
  $$ IoU = \frac{A \cap B}{A \cup B} = \frac{TP}{TP + FP + FN} $$
- ถ้า BG มี precentage เยอะก็ตอบให้เป็น BG หมดก็จบ -> IoU ก็โอเค เพราะ FP,FN มันมีนิดเดียว
### Dice Score

- คล้าย IoU แต่ให้ความสำคัญ **คำตอบที่ถูก** มากขึ้น เพิ่ม weight ทำให้ได้คะแนนสูง
- สูตร:
  $$ Dice = \frac{2TP}{2TP + FP + FN} $$
- ในงาน segmentation มักใช้ **Dice loss = 1 – Dice score**  
  → บังคับให้ mask ทับกันเยอะ ๆ

**ทั้ง IoU และ Dice สุดท้ายจะใช้เป็น mean ของการกำหนด class เป็น Positive, หรือเอาเฉพาะอันที่สนใจเช่น object = positive**
### Pixel Accuracy / Precision / Recall / F1

- **Pixel accuracy (Acc)** = จำนวน pixel ที่ทายถูก / pixel ทั้งหมด 
    - แต่ถ้า class imbalance (BG เยอะ) → Acc สูงแต่ไม่ได้ดี
- ใช้ **Precision / Recall / F1** ต่อ class แล้ว
    - Macro-F1 = average ระหว่าง class (ทุก class สำคัญเท่ากัน) บวกแล้วหารด้วยจำนวน class
    - Micro-F1 = นับรวมทุก pixel ก่อนคำนวณ (class ใหญ่มีน้ำหนักเยอะ) คูณด้วย pixel บวกแล้วหารด้วยจำนวน pixel

---

## FCN – Fully Convolutional Networks

- เปลี่ยนจาก CNN แบบ classification (ลงท้ายด้วย FC layer)  
  → เป็น network ที่มีแต่ conv layer
- Idea:
    - ใช้ CNN encoder ทำให้ feature map เล็กลง
    - ใช้ **transposed convolution (ConvTranspose2d)** หรือ upsampling  
      เพื่อขยายกลับไปที่ resolution เดิม, มี Skip Connection มาช่วยด้วย
        - FCN-32s : จาก 1x1 (A) -> 32x32 เลย TransConv รอบเดียว
        - FCN-16s : เอา A -> 2x2 ก่อน เอามาบวกกับ 2x2 ของตอน encoder (skip connection) ได้ 2x2 (B) -> 32x32
        - FCN-8s  : เอา B -> 4x4 ก่อน เอามาบวกกับ 4x4 ของตอน encoder (skip connection) ได้ 4x4 -> 32x32
        - Result : 8s > 16s > 32s
- **Transposed convolution** : ทำงานยังไงดู pg.17
    - มี parameter: kernel size, stride, padding
    - มองได้เหมือน “inverse” ของ conv ปกติ
    - ใช้ stride > 1 เพื่อ upsample feature map
    - padding -> คือตัดขอบของ result ออก
---

## U-Net
- **Encoder–Decoder U-shape + Skip Connections**
    - Encoder: ลด spatial size, เพิ่ม channels 
    - Decoder: ขยาย spatial size กลับ, ลด channels (สุดท้ายเหลือเท่ากับจำนวน class แต่ละ depth บอก prob ของ class นั้นๆ)
    - มี **skip connection** เอา feature จาก encoder มาต่อ (concat) กับ decoder  
      → ช่วยให้เก็บ fine details (boundary)

---

## Loss Functions for Segmentation

- ใช้ **สอง loss รวมกัน** เพื่อให้ training stable:
    - **DiceLoss (1 - Dice-Score) : focus pixel**
        - เน้นให้ overlap ระหว่าง prediction mask กับ GT สูง
        - ดีมากเวลา class positive น้อย (object เล็ก ๆ)
    - **BCEWithLogitsLoss : focus class**
        - รวม Sigmoid + Binary Cross Entropy ในตัวเดียว เสถียรกว่าใช้ Sigmoid แล้วค่อย BCE แยกกัน
        - สามารถใส่ class weight เพื่อปรับ balance precision/recall ได้

---

## DeepLab By Google

### DeepLab v1

- ใช้ **DCNN + Atrous (Dilated) Convolution**
- Dense Coord-Convolution Network (DCNN) : เพิ่ม Coordinate (x,y) ลงใน input 
- Atrous conv:
    - ใส่ “ช่องว่าง” ใน kernel → receptive field ใหญ่ขึ้น
    - ไม่ต้องเพิ่ม kernel size และไม่ต้อง down-sampling 
- Output feature map upsample กลับไปขนาดภาพเดิมด้วย bilinear interpolation -> ไวและไม่ต้อง Train เหมือน TransposeConvo
- ใช้ **fully connected CRF (Conditional Random)** ช่วย refine ขอบ object ให้คมขึ้น
    - เป็น discriminative model → สนใจตรง ๆ เลยว่า
ถ้าเห็น input แบบนี้ อยากให้ label ออกมาเป็นอะไร
### DeepLab v2 – ASPP (Atrous Spatial Pyramid Pooling)

- ใช้ **หลาย ๆ atrous conv ขนาดเท่ากัน แต่ dilation rate ต่างกัน** ทำงานขนานกัน
- เพิ่ม Global Average Pooling (context ใหญ่ ๆ) แล้ว upsample
- Concatenate ทุก branch ตาม channel dimension → ได้ feature รวมหลาย scale

### DeepLab v3 / v3+

- Encoder–Decoder Model คล้าย U-Net → able to sharp object boundaries
- ใช้ **depth-wise separable convolution** เพื่อให้คำนวณไวขึ้น
    - depth-wise separable conv ตามด้วย 1x1 pointwise conv (1x1 ConVo) to increase computational efficiency
- v3+:
    - ใช้ Aligned Xception แทน ResNet101 เป็น encoder (ลึกและประสิทธิภาพกว่า)
    - **แทน max pooling ทั้งหมดด้วย depth-wise separable conv**

---

## Depth-wise Separable Convolution

- แยก Conv ปกติออกเป็น 2 ขั้น:
    1. **Depth-wise convolution**: filter แยกต่อ channel
        - input C channels → apply C filters, อย่างละ k×k
    2. **Point-wise (1×1) convolution**
        - 1×1 conv เพื่อ mix channels และกำหนดจำนวน output channels
- เทียบจำนวน parameters:
    - Standard conv: `in_c * out_c * k * k` (เช่น 3x5x5x32 = 2400)
    - Depth-wise + 1×1: `(in_c * k * k) + (in_c * out_c)` (75+96=171)
- สรุป: **ถูกและเบา** → เหมาะกับ mobile network (MobileNet, Xception)

---

## EfficientNet-NAS-FPN, HRNet (ภาพรวมสั้น ๆ)

- **EfficientNet-NAS-FPN**
    - Backbone = EfficientNet (compound scaling → balance depth/width/resolution)
    - FPN (Feature Pyramid Network) รวม feature multi-scale
    - ใช้ NAS หา architecture FPN ที่ลงตัวกับ backbone → ดีทั้ง accuracy + efficiency
- **HRNet (High-Resolution Network)**
    - เก็บ high-resolution feature ตลอด network (ไม่ใช่ downsample อย่างเดียว)
    - มี parallel branches หลาย resolution แล้ว fuse กันหลายครั้ง
    - ดีมากสำหรับงานที่ต้องการ boundary ละเอียด เช่น segmentation / pose estimation

---

