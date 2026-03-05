# Yoga Posture Detection using YOLO + MediaPipe + OpenCV (Python)

This project detects **people**, tracks full-body **pose landmarks**, draws **joints + skeleton lines**, and classifies basic **yoga postures** in real time.

## Features

- Real-time person detection from webcam (YOLO)
- Full-body keypoint and skeleton visualization (MediaPipe)
- Yoga posture classification (Tree, Raised Arms, Warrior II, T Pose, Neutral)
- Video file input support
- Bounding boxes + confidence labels
- Live person count per frame
- Optional output video saving

## Project Structure

```
CP/
├── requirements.txt
├── README.md
└── src/
  ├── engine.py
  ├── humanDetector.py
  └── main.py
```

## Setup

### 1) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run

### Webcam (default)

```bash
python src/main.py
```

### Specific webcam index

```bash
python src/main.py --source 0
```

### Video file input

```bash
python src/main.py --source /path/to/video.mp4
```

### Save annotated output

```bash
python src/main.py --source /path/to/video.mp4 --save --output outputs/result.mp4
```

## Controls

- Press `q` to quit.

## Troubleshooting

- If app starts and closes immediately, try another webcam index:

```bash
python src/main.py --source 1
```

- On macOS, allow **Camera** access for Terminal/VS Code in:
  - System Settings → Privacy & Security → Camera

- To tolerate temporary camera hiccups longer, increase empty-frame limit:

```bash
python src/main.py --max-empty-frames 120
```

## Notes

- First run may download YOLO weights (`yolov8n.pt`) and MediaPipe model (`pose_landmarker_lite.task`) automatically.
- To improve accuracy, try a larger model (for example: `--model yolov8s.pt`).
