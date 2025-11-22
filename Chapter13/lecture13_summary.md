
# Lecture 13 – Generative Models for Images

## ภาพรวมสาย Image Generation

สามตระกูลหลักของ deep generative model

- **VAE (Variational Autoencoder)** – 2013  
  encoder–decoder + probabilistic latent space
- **GAN (Generative Adversarial Network)** – 2014  
  generator vs discriminator เล่นเกมกัน
  - PixToPix (2017)
  - Style GAN (2018)
- **Diffusion Models** – 2020–now  
  ค่อย ๆ เติม noise แล้วค่อย ๆ ล้างออก

ตอนนี้ “diffusion” เป็นตัวหลักของ text-to-image รุ่นใหม่ ๆ (DALL·E 2, Midjourney, Sora ฯลฯ)

---

## VAE

โครงสร้าง: encoder–decoder เหมือน autoencoder แต่เป็นแบบ probabilistic

- Encoder:
  - รับภาพ x แล้วแปลงเป็นพารามิเตอร์ของ Gaussian: μ(x), log σ²(x)
  - นิยาม q(z|x) = N(μ(x), σ²(x) I)
  - ใช้ reparameterization trick: z = μ + σ ⊙ ε, ε ~ N(0, I)
- Decoder:
  - รับ z แล้ว reconstruct ภาพ ŷ (approx x)

### Loss

- Reconstruction loss: วัดความต่างระหว่าง x กับ ŷ (เช่น MSE, L1, บางงานใช้ SSIM)
- KL divergence: D_KL(q(z|x) || N(0, I)) เพื่อบังคับให้ latent ใกล้ Gaussian prior

รวม ๆ:   L = ReconLoss(x, ŷ) + KL(q(z|x) || N(0, I))

### การ generate ภาพ

- ตอน generate ไม่ต้องมีภาพ x
- สุ่ม z ~ N(0, I) จาก prior แล้วส่งเข้า Decoder → ได้ภาพใหม่ที่สมเหตุสมผลใน data manifold

**ปรับปรุงที่เจอบ่อย**

- **CVAE (Conditional VAE)**  
  ใส่ condition y (เช่น class, text) เข้าที่ encoder + decoder  
  → คุมได้ว่าอยากให้ generate class ไหน / ตาม text อะไร, จาก VAE ที่สุ่มได้อย่างเดียว
- **alignDRAW**  
  1st modern text-to-image model, ใช้ recurrent VAE + attention กับ text sequence as input
- **IntroVAE**  
  Self-evaluate the quality of generated samples and subsequently  self-improve (similar to GAN)

---

## GAN 

### Concept

มีสอง network เล่นเกมกัน:

- **Generator G(z)** – Decoder, รับ noise z → สร้างภาพปลอม, คล้าย VAE
- **Discriminator D(x)** – Encoder, ดูว่าภาพเป็น real(จาก data) หรือ fake(จาก G), classification model

### Training
เกม min–max:
- เมื่อ 0 = fake, 1 = real
- D พยายามแยก real/fake ให้ดี → maximize log D(real) + log(1−D(fake))
- G พยายามหลอก D → minimize log(1−D(fake)) หรือ maximize log D(fake), recommend maximize log D(fake) เพราะไม่งั้น D กับ G จะมีจุดประสงค์ชนกัน

D(x) ใช้ loss ส่วนใหญ่เป็น **binary cross-entropy**

### ใช้ทำอะไร

- สร้าง “สิ่งที่ไม่เคยมีอยู่จริง”
  - face synthesis, style transfer, image-to-image, inpainting
  - restoration, super-resolution ฯลฯ
- **Data augmentation** 
  - Create Dataset

สไลด์มีตัวอย่าง:
- ซ่อมรูปเก่า
- เปลี่ยนสภาพอากาศ, เปลี่ยน style ภาพ landscape
- เปลี่ยน resolution, inpainting รูปที่ขาดหาย

---

## DCGAN – Architectures

### Generator

- Input: noise z (เช่น ขนาด 128)
- Stack ของ:
  - `ConvTranspose2d` (upsample)
  - `BatchNorm2d`
  - `ReLU`
- ค่อย ๆ ขยาย spatial size: 1×1 → 4×4 → 8×8 → … → 64×64
- เลเยอร์สุดท้ายใช้ **Tanh** → output อยู่ในช่วง [-1, 1]

### Discriminator

- Input: ภาพ `nc × 64 × 64`
- Stack ของ:
  - `Conv2d` (downsample)
  - `BatchNorm2d`
  - `LeakyReLU`
- ค่อย ๆ ลด spatial size, เพิ่ม channels
- เลเยอร์สุดท้ายให้ scalar probability ผ่าน Sigmoid

### Training loop (โดยย่อ)

1. อัปเดต D:
    - ใช้ real images (label=1) → loss_real
    - ใช้ fake จาก G(z) (label=0) → loss_fake
    - รวมเป็น `errD` → backward + `optimizerD.step()`
2. อัปเดต G:
    - สร้าง fake ใหม่จาก noise
    - ป้อนเข้า D แต่ใช้ label=1 (อยากให้ D คิดว่าเป็น real)
    - ได้ `errG` → backward + `optimizerG.step()`

สไลด์มีตัวอย่าง **real cat faces vs fake cat faces** จาก DCGAN

---

## Evolution & Variants ของ GAN

### ProGAN – Progressive Growing

- Train จากภาพ **ความละเอียดต่ำ → สูงทีละขั้น**
  - เริ่มที่ 4×4 → ค่อยเพิ่มเลเยอร์จนถึง 1024×1024
- Stable ขึ้นในการ train รูปใหญ่ ๆ

### CGAN – Conditional GAN

- เพิ่ม condition y เข้า G และ D (เช่น class label, text, image อื่น)
- ทำให้ควบคุมสิ่งที่ generate ได้ เช่น “แมว”, “หมา”, “รองเท้า”

### Pix2Pix – Image-to-Image Translation

โจทย์: มีคู่ภาพ (xA, xB) เช่น

- segmentation mask → ภาพจริง
- แผนที่ ↔ aerial photo
- sketch → photo
- B/W → color
- รูปที่ mask ทิ้งบางส่วน → inpaint ภาพเต็ม

หลัก ๆ:

- ใช้ **PatchGAN discriminator**:
  - D ให้ output map ของ patch-scale “real/fake” แทน scalar เดียว
  - เน้น local realism (texture, edge) มาก
- Loss = **GAN loss + λ · L1 loss**:
  - GAN loss: ให้ภาพดู real
  - L1(xB, G(xA)): ให้ผลลัพธ์ใกล้ target pixel-wise, ลด blur (L1 ดีกว่า L2 สำหรับรายละเอียดภาพ)

---

## ข้อจำกัดของ GAN

- **Mode collapse**  
  G สร้างรูปแค่บางแบบ ไม่ครอบคลุมความหลากหลายของ data
- **Unstable training**  
  ถ้า D แรงเกิน G หรือ learning rate เละ → gradient หาย / แกว่ง
- ยากในการ model distribution ที่หลากหลายมาก ๆ แบบครบทุก mode

ปัญหาเหล่านี้เป็นหนึ่งในเหตุผลที่คนเริ่ม shift ไปทาง diffusion models

---

## Diffusion Models – DDPM

### Concept หลัก

มีสองขั้นตอน:

1. **Forward (noising) process**
    - เริ่มจากภาพจริง \(x_0\)
    - ค่อย ๆ เติม Gaussian noise ได้ \(x_1, x_2, ..., x_T\)
    - สุดท้าย \(x_T\) จะกลายเป็น noise เกือบสมบูรณ์แบบ (high entropy)
2. **Reverse (denoising) process**
    - เริ่มจาก noise \(x_T\)
    - ค่อย ๆ ใช้ model (เช่น U-Net) ทำนายและลบ noise ทีละ step
    - ไปจนถึง \(x_0\) → เป็นภาพใหม่ที่ sample มาจาก data distribution

### การ train

- ให้ model รับ (x_t, t) และ predict noise ε ที่ถูกเติมใน step นั้น
- ใช้ **MSE loss** ระหว่าง noise จริง vs noise ที่โมเดลทำนาย
- สุ่ม t หลาย ๆ level → โมเดลเรียนรู้การ denoise ทุกระดับ noise

---

## Latent Diffusion & Stable Diffusion

**Stable Diffusion** ใช้ไอเดีย **Latent Diffusion Model (LDM)**:

- ใช้ autoencoder บีบภาพให้เป็น latent z
- ทำ diffusion ใน latent space แทน pixel space
  - ลดความซับซ้อน, ประหยัด compute
- ใช้ U-Net เป็น core denoiser
- ใส่ text condition ผ่าน text encoder (เช่น Transformer) + cross-attention
  → ได้ text-to-image ที่ powerful และยังรันบน GPU ปกติได้

---

## Diffusion vs GAN

- **Diffusion models**
  - Train เสถียรกว่า GAN มาก
  - ไม่ค่อยเจอ mode collapse
  - Flexible: ใช้กับภาพ, เสียง, วิดีโอ, 3D ฯลฯ
  - แต่ sampling ช้า (ต้องหลาย step)
- **GAN**
  - Sampling เร็วมาก (แค่ forward ทีเดียว)
  - แต่ train ยาก, ปัญหา mode collapse/unstable เยอะ

แนววิจัยใหม่ ๆ เลยพยายาม:
- ทำ diffusion ให้ “few-step / one-step” เข้าใกล้ความเร็ว GAN
- หรือ hybrid แนว diffusion+GAN

---

## Future Directions (จาก slide)

- **Few-step / one-step generation**
  - ลดจำนวน step ใน reverse process
- **Efficient architectures**
  - ปรับ backbone / top-layers ให้เบาแต่คุณภาพดี
- **Complex lighting / shadows**
  - ให้โมเดล handle แสงเงาซับซ้อนได้ดีขึ้น
- **Robust editing**
  - ปรับแก้ภาพโดยไม่ทำลาย context อื่น ๆ
- **Better evaluation metrics**
  - อยากได้ metric ที่ดู semantic similarity มากกว่า pixel error (PSNR/SSIM)

---

## Ethics & Social Impact

- **Deepfakes & misinformation** – ปลอมหน้า/เสียง/วิดีโอได้เหมือนมาก
- **Consent & privacy** – ใช้หน้าคน/รูปคนโดยไม่ขออนุญาต
- **Bias & discrimination** – ถ้า data ลำเอียง output ก็ลำเอียงตาม
- **Art & IP** – ปัญหาเรื่อง style ของศิลปิน, ลิขสิทธิ์, creative ownership
- **Economic impact** – กระทบงานสาย creative บางส่วน แต่ก็เปิดโอกาสงานใหม่ในอีกด้าน

