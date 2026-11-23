### Real-Time Object Detection & Spatial Recognition Platform
**Edge-Optimized YOLOv8 Architecture, Multi-Task Loss Formulations & OpenCV Interface**  
<sub>`Python` • `YOLOv8` • `OpenCV` • `PyTorch` • `MSc in AI & Deep Learning (UNIWA)`</sub>

---

This repository contains the software implementation, computer vision pipelines, and mathematical derivations developed for the **Platforms for AI and Python Programming (AIDL_03)** course in the **MSc in Artificial Intelligence and Deep Learning** at the **University of West Attica**.

The project develops a high-throughput, low-latency real-time object detection system using an anchor-free **YOLOv8 Nano** backbone coupled with a customized **OpenCV Heads-Up Display (HUD)** for interactive inference.

---

## 📑 Project Structure
```text
├── app.py                                         # Real-time inference application with live HUD & controls
├── Fotis_Papadopoulos_YOLOv8_Object_Recognition.pdf # Academic technical report with full mathematical proofs
├── screenshots/                                   # Automatically captured raw and annotated frame pairs
└── README.md                                      # Project overview and architectural documentation
```

---

## 🔬 Theoretical Foundations & Computer Vision Architecture

### 1. Convolutional Operators & Algorithmic Commutativity
- **Mathematical 2D Convolution vs. Cross-Correlation:** Explores the discrete sliding-window formulations, contrasting standard flipped-kernel convolution with deep learning cross-correlation.
- **Computational Complexity Proof:** Mathematically demonstrates how exploiting the commutative property of convolution reduces local loop algorithmic complexity from image-scale $O(H \cdot W)$ down to kernel-scale $O(K_H \cdot K_W)$ per output pixel.
- **Multi-Channel Tensor Convolutions & SiLU Activation:** Formulates forward passes across 3D input tensors with learnable bias broadcasting and element-wise Sigmoid Linear Unit ($\text{SiLU}(x) = x \cdot \sigma(x)$) non-linear gating.

### 2. YOLOv8 Anchor-Free Decoupled Architecture
- **Coordinate Transformations:** Formulates linear mappings between Center-based $(x_c, y_c, w, h)$ regression formats and Corner-based $(x_1, y_1, x_2, y_2)$ rendering spaces with spatial normalization.
- **Multi-Scale Detection Pyramids:** Preserves small features and semantic contexts across three downsampling strides:
  - $P_3$ (Stride 8): High spatial resolution for small objects.
  - $P_4$ (Stride 16): Intermediate scale representation.
  - $P_5$ (Stride 32): High receptive field for large, dominant objects.
- **Decoupled Detection Head:** Separates the classification branch ($C=80$ COCO classes) from the bounding box regression branch ($4 \times 16$ distribution bins via DFL), mitigating task-conflict interference during gradient updates.

### 3. Multi-Task Loss Functions
Optimization minimizes a unified loss: $\mathcal{L}_{\text{total}} = 7.5\mathcal{L}_{\text{box}} + 0.5\mathcal{L}_{\text{cls}} + 1.5\mathcal{L}_{\text{dfl}}$
- **Complete Intersection over Union (CIoU) Loss:** Combines overlap area, normalized center point Euclidean distance penalty ($\frac{\rho^2}{c^2}$), and dynamic aspect ratio consistency ($\alpha v$).
- **Distribution Focal Loss (DFL):** Models spatial boundary uncertainty as a continuous probability distribution over 16 discrete bins around ground-truth offsets.
- **Binary Cross-Entropy (BCE) Loss:** Provides multi-label classification probabilities without mutual exclusivity constraints.

---

## 💻 Real-Time Application Features (`app.py`)

- **Edge-Optimized Inference:** Powered by Ultralytics YOLOv8 Nano (`yolov8n.pt`), pre-trained on the COCO dataset (80 categories), delivering 30+ FPS real-time processing on standard CPU architectures.
- **Cyberpunk-Themed Live HUD:**
  - Real-time **FPS & Latency (ms)** hardware diagnostics.
  - Live object counter (`num_detections in view`).
  - Active system health indicator.
- **Interactive Dual-Frame Capture:**
  - Pressing `[S]` captures and saves both the **raw unaltered camera frame** and the **annotated detection frame** simultaneously with a visual screen flash confirmation.
  - Pressing `[Q]` safely releases hardware resources and terminates execution.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install ultralytics opencv-python numpy
```

### Running the Application
```bash
python app.py
```
