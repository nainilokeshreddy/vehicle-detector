# AI-Based Autonomous Vehicle Perception System

An educational, local Deep Learning and Computer Vision mini project designed to demonstrate real-time perception for self-driving vehicles.

## 📌 Project Overview
Autonomous vehicles rely on perception systems to sense and interpret their surrounding road environment. This system processes dashboard camera footage or static images to extract three critical perception layers:
1. **Road Object Detection:** Identifies cars, trucks, buses, motorcycles, bicycles, and pedestrians.
2. **Lane Detection:** Identifies ego-lane boundaries to support lane-keeping assistance.
3. **Traffic Sign Recognition:** Detects and classifies regulatory and warning road signs.
4. **Combined Perception Pipeline:** Simultaneously displays all perception layers on a single video/image feed with clear overlays and statistics.

---

## 📁 Project Structure

```text
autonomous_vehicle_ai/
│
├── screenshots/               # Visual outputs, sample detections, and viva demo figures
├── dataset/                   # Dataset directory
│   ├── images/               # Raw test/train images
│   ├── labels/               # Ground truth annotations
│   └── README.md             # Dataset documentation
├── models/                    # Stored deep learning model weights (.pt, .onnx)
├── outputs/                   # Processed outputs
│   ├── images/               # Saved perception result images
│   └── videos/               # Saved perception result video recordings
├── src/                       # Core python source modules
│   ├── object_detection.py    # Vehicle & road user detection module
│   ├── lane_detection.py      # Lane detection & segmentation module
│   ├── traffic_sign_detection.py # Sign detection & classification module
│   ├── video_processing.py    # Video frame streaming & rendering
│   └── utils.py               # Helper visualization & I/O utilities
├── main.py                    # Main executable entry point with CLI options
├── requirements.txt           # Python dependencies list
├── README.md                  # Project documentation
└── project_report.md          # College mini-project viva and documentation report
```

---

## 🚀 Execution Modes (Planned)

| Mode | Command Example | Description |
| :--- | :--- | :--- |
| **Combined** | `python main.py --mode combined --source-type video --input sample.mp4` | Full perception pipeline with objects, lanes, and signs |
| **Objects** | `python main.py --mode objects --source-type image --input road.jpg` | Vehicle and pedestrian detection |
| **Lanes** | `python main.py --mode lanes --source-type video --input road.mp4` | Road lane boundary detection |
| **Signs** | `python main.py --mode signs --source-type image --input sign.jpg` | Traffic sign identification |

---

## 🎓 College Viva & Academic Highlights
- **Perception vs. Planning vs. Control:** Perception answers *"What is around the car?"*, planning answers *"What should the car do?"*, and control executes steering/throttle/braking.
- **Edge Computing & Local Execution:** The system runs completely locally without external cloud dependencies, representing on-board vehicular computing.
- **Modularity:** Each perception task is encapsulated in its own module inside `src/`, making the codebase easy to inspect, debug, and explain in an oral exam.
