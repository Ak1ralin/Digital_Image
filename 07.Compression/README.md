# Image Compression
---

## Objectives
- เข้าใจ **data, information, redundancy** ในภาพ  
- อธิบายเทคนิค **lossless และ lossy compression**  
- เข้าใจมาตรฐาน **JPEG**【95†source】  

---

## Core Concepts

### 1. Fundamentals
- Image compression = ลดขนาดไฟล์ โดยยัง retain ข้อมูลที่จำเป็น  
- Compression ratio:  

   $ C_R = \frac{ori}{compress} $ - can be any information-carrying unit  
- Relative data redundancy

   $ R_D = 1 - \frac{1}{C_R} $

- ประเภท:  
  - **Lossless compression** → exact reconstruction, ratio 2–10, 100% reconstruct, 
      - medical & business, satellite  
  - **Lossy compression** → ข้อมูลสูญหายบางส่วน, ratio 10–50, ใช้ในภาพทั่วไป

- Some definition :
   - uncompressed (original) --compress--> compressed --reconstruct--> decompressed 
   - Data : raw data, might redundancy (give same info) so we reduce it
   - Information : interpretation of data in meaningful way
- Redundancy types : root cause
  1. Coding redundancy : represent data in a wrong way 
      - Symbol Encoder : พยายามทำให้ code optimal = เข้าใกล้ entropy 
      - $L_{avg} = \sum_{k=0}^{L-1} l(r_k)p_r(r_k)$
         - ใช้บอกว่า coding นี้ใช้กี่บิทต่อ pixel, เข้าใกล้ entropy คือดี ยิ่งน้อยยิ่งดี
         - $l(r_k)$ จำนวนบิทที่ใช้ในการแสดง $r_k$
         - $p_r(r_k)$ โอกาสที่ $r_k$ จะปรากฎ
  2. Interpixel redundancy : adjacent pixel tend to be highly correlated
      - Mapper : ใช้ different แทนบอกค่าตรงๆ
  3. Psychovisual redundancy : fully detailed in unnessary part  
      - Quantizer : ลด resolution => lossy ถ้ามีคือ lossy compression
---

### 2. Elements of Information Theory
- Information of event E:  

$ I(E) = -\log P(E) $  

- Entropy/uncertainty of source:  

$ H(z) = - \sum p(z_k) \log_2 p(z_k) $  
   - log2 ได้ผลลัพธ์หน่วยเป็น bit

- Entropy ≈ minimum average bits/pixel needed

---

### 3. Image Compression Models
General model:  
Source encoder → Channel encoder → Transmission → Channel decoder → Source
- Components:  
  - Mapper  
  - Quantizer : for lossy compression only
  - Symbol encoder  
  - Inverse mapper

---

### 4. Error-Free (Lossless) Compression
1. **Variable-Length Coding**  
   - Huffman coding: สร้าง tree จาก histogram probability
      - น้อยสุดสองอันรวมกันไปเรื่อยๆ จนเหลือแค่ 2 
      - ตอนกำหนด bit : มากกว่าเติม 0, น้อยกว่าเติม 1   
   - Arithmetic coding: encode เป็นเลข real [0,1), ใช้ distribution  

2. **LZW Coding**  
   - ใช้ dictionary-based compression, ใช้ 9 bits 256 เป็นต้นไปใช้เก็บ pattern ใหม่ใน dict  
   - ใช้ใน GIF, TIFF, PDF

3. **Bit-plane Coding**  
   - แยก plane ของ bits MSB = เห็นภาพสุด, ใช้ run-length coding  

4. **Run-Length Coding (RLC)**  
   - ดีมากกับ binary images, encode (value, length)
   - รวมเป็นก้อนๆ สำหรับ pixel ที่มีค่าเท่ากัน
   - binary horizontal : มีแค่จำนวนก็พอ เริ่มด้วย 0
   - extended : (color, #run) : ใช้เป็น tuple pair

5. **Lossless Predictive Coding**  
   - ใช้ค่าคาดเดาจากเพื่อนบ้าน → encode error  

---

### 5. Lossy Compression
1. **Lossy Predictive Coding**  
   - เพิ่ม quantizer → บีบ error เข้าช่วงจำกัด  

2. **Transform Coding**  
   - ใช้ DCT (Discrete Cosine Transform) → JPEG baseline  
      - เนื่องจากใช้เฉพาะ real number เลยเป็น lossy
   - ใช้ Wavelet → JPEG2000  

**JPEG Steps:**  
- Level shift pixels (−128)  
- DCT (8×8 block)  
- Quantization (lossy step)  
- Zig-zag scan (DC, AC coefficients)  
- Entropy coding (Huffman/Arithmetic)  

---

### 6. Standards Summary
- **JPEG**: DCT, quantization, Huffman, 8×8 blocks  
- **JPEG2000**: wavelet, lossless+lossy, arithmetic coding  
- **PNG**: DEFLATE (Huffman + LZ77)  
- **BMP/TIFF**: LZW, RLC  
- **Predictive Coding**: DPCM (multimedia bandwidth reduction)  

---
## Summary
- Compression = ลด redundancy → save storage, bandwidth  
- Lossless (Huffman, Arithmetic, LZW, RLC, Predictive)  
- Lossy (Transform coding, Quantization, Predictive)  
- JPEG = DCT + quantization + Huffman + zig-zag coding  
- JPEG2000 = wavelet + arithmetic coding  
- Standards = JPEG, JPEG2000, PNG, TIFF, BMP【95†source】  
---
##  Python Example (Huffman via `heapq`)
```python
import heapq
from collections import Counter, namedtuple

class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class Leaf(namedtuple("Leaf", ["char"])):
    def walk(self, code, acc):
        code[self.char] = acc or "0"

def huffman_encode(s):
    h = []
    for ch, freq in Counter(s).items():
        h.append((freq, len(h), Leaf(ch)))
    heapq.heapify(h)
    count = len(h)
    while len(h) > 1:
        freq1, _c1, left = heapq.heappop(h)
        freq2, _c2, right = heapq.heappop(h)
        heapq.heappush(h, (freq1 + freq2, count, Node(left, right)))
        count += 1
    code = {}
    if h:
        [(_freq, _count, root)] = h
        root.walk(code, "")
    return code

# Example usage
s = "ABBCCCDDDDEEEEE"
code = huffman_encode(s)
print(code)
```

---

