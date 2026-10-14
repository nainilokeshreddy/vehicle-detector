# AI-Based Autonomous Vehicle Perception System: Project Report

## Academic Mini-Project Documentation

---

### 1. Abstract
Autonomous vehicles rely on robust perception systems to interpret complex driving scenes in real time. This mini-project demonstrates a local, Python-based autonomous vehicle perception pipeline combining Deep Learning and Computer Vision. The system addresses three fundamental tasks: (1) multi-class road obstacle detection (vehicles and pedestrians), (2) road lane detection for lateral guidance, and (3) traffic sign recognition for regulatory compliance. Operating entirely locally, this project provides a clear modular architecture ideal for academic demonstration and viva presentations.

---

### 2. Introduction & Problem Statement
Perception is the initial stage of any autonomous driving stack, preceding trajectory planning and vehicle control. In real-world driving environments, lighting variations, road curvature, and high obstacle density present significant challenges. 

**Objectives:**
- Implement real-time multi-class object detection for road obstacles.
- Implement robust lane detection to locate ego-vehicle boundaries.
- Recognize critical road traffic signs.
- Provide unified, frame-by-frame visualization for both static images and video feeds.
- Produce structured outputs in designated directories.

---

### 3. System Architecture & Workflow

```text
       [Input: Image / Video Stream]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 ┌───────────────┐       ┌───────────────┐
 │ Deep Learning │       │Computer Vision│
 │   Detector    │       │Lane Extractor │
 │ (Objects/Signs│       │ (Canny/Hough/ │
 │    YOLO/CNN)  │       │  Polynomial)  │
 └───────┬───────┘       └───────┬───────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Perception Fusion HUD │
         │   Overlay & Metrics   │
         └───────────┬───────────┘
                     ▼
        [Output: Image / Video File]
```

---

### 4. Modules Breakdown
1. **Object Detection (`src/object_detection.py`):** Uses convolutional neural networks to identify and localize dynamic road entities.
2. **Lane Detection (`src/lane_detection.py`):** Employs edge detection, region of interest masking, and polynomial curve fitting to delineate driving corridors.
3. **Traffic Sign Detection (`src/traffic_sign_detection.py`):** Isolates and classifies road signs into warning, regulatory, and guide categories.
4. **Video Processing (`src/video_processing.py`):** Encapsulates stream acquisition, frame buffering, inference loop, and synchronized output rendering.
5. **Utilities (`src/utils.py`):** Reusable visualization primitives (bounding boxes, overlays, text rendering).

---

### 5. Viva Voce Preparation (Key Questions & Answers)
- **Q1: What are the three core layers of an autonomous vehicle software stack?**
  *Answer:* Perception (sensing and understanding the world), Planning (behavioral decision-making and trajectory planning), and Control (actuator commands for steering, acceleration, and braking).
- **Q2: Why is deep learning preferred over traditional computer vision for object detection?**
  *Answer:* Deep learning models learn hierarchical feature representations directly from data, making them significantly more invariant to scale, viewpoint, occlusion, and lighting changes than handcrafted features (like HOG or Haar cascades).
- **Q3: What is the role of lane detection in driver assistance systems (ADAS)?**
  *Answer:* Lane detection is the backbone of Lane Departure Warning (LDW) and Lane Keep Assist (LKA) systems, ensuring the car remains centered within its designated lane.

---

### 6. Next Implementation Steps
- [ ] Setup Python environment with specified dependencies.
- [ ] Acquire sample test media (road images and driving video snippets).
- [ ] Implement object detection pipeline using pretrained weights.
- [ ] Implement lane boundary extraction algorithm.
- [ ] Implement traffic sign classification module.
- [ ] Integrate into the unified perception HUD and verify local outputs.
