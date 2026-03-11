# Face Recognition System

Real-time face recognition from a webcam using [DeepFace](https://github.com/serengil/deepface) and OpenCV.
Detects faces, matches them against a database of known people, and displays their name on screen.

---

## Features

- Real-time webcam face detection and recognition
- Supports multiple people with multiple photos each
- Displays name in green for known faces, red for unknown
- Saves one snapshot per person per session
- Background threading — video stream never freezes
- Frame skipping and result caching for performance

---

## Setup

**Requirements:** Python 3.10–3.12

```bash
pip install -r requirements.txt
```

---

## Face Database

Create a folder for each person inside your database directory:

```
test/
├── AYA/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── Sara/
│   └── 1.jpg
└── Ahmed/
    ├── 1.jpg
    └── 2.jpg
```

- Folder name = the name displayed on screen
- 5–10 clear, front-facing photos per person recommended

---

## Usage

```bash
py -3.12 main.py
```

- Press **Q** or **ESC** to quit

---

## Output

Snapshots are saved automatically to:

```
C:/Users/<you>/face_reco_output/
├── recognized/   ← known people
└── unknown/      ← unrecognized faces
```

One image per person is saved per session.

---

## Configuration

Edit the bottom of `main.py` to change settings:

```python
custom_stream(
    db_path          = "path/to/your/database",  # face database folder
    model_name       = "ArcFace",                # recognition model
    detector_backend = "opencv",                 # face detector
    source           = 0,                        # webcam index
)
```
