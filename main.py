from deepface import DeepFace
import cv2
import os
import threading
import time
from datetime import datetime

# Create folders to save the recognized and unknown faces
recognized_folder = "known_faces"
unknown_folder = "unknown_faces"
os.makedirs(recognized_folder, exist_ok=True)
os.makedirs(unknown_folder, exist_ok=True)

# ── EDIT 1: Frame skipping ─────────────────────────────────────────────────────
# Only run recognition every N frames instead of every frame.
# Reduces CPU load significantly. Increase FRAME_SKIP for more speed.
FRAME_SKIP = 4

# ── EDIT 2: Result cache ───────────────────────────────────────────────────────
# Stores the last known label + box per face position so we can draw results
# on skipped frames too. Avoids flickering between frames.
# Format: { "x_y_w_h": {"label": str, "color": tuple, "timestamp": float} }
face_cache = {}
CACHE_EXPIRY = 3.0  # seconds before a cached result is considered stale

# ── EDIT 3: Save throttle ──────────────────────────────────────────────────────
# Tracks the last time we saved an image per identity to avoid disk flooding.
# Format: { identity: last_save_timestamp }
last_save_time = {}
SAVE_INTERVAL = 3.0  # seconds between saves per identity

# ── EDIT 4: Background recognition thread ─────────────────────────────────────
# Recognition runs in a separate thread so the video loop never blocks.
# The main loop just reads results from shared state.
recognition_lock = threading.Lock()
pending_frame = None       # frame queued for recognition
recognition_running = False


def save_frame(frame, identity, folder):
    """ Saves the frame with a unique timestamp — throttled per identity. """
    now = time.time()
    # ── EDIT 3 applied: skip save if we saved this identity recently ──────────
    if now - last_save_time.get(identity, 0) < SAVE_INTERVAL:
        return
    last_save_time[identity] = now

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{folder}/{identity}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved: {filename}")


def recognition_worker(frame, db_path, model_name, detector_backend):
    """
    Runs in a background thread.
    Detects faces, runs recognition, and updates face_cache.
    ── EDIT 4 applied ──────────────────────────────────────────────────────────
    """
    global recognition_running
    try:
        # ── EDIT 6: Confidence filter ──────────────────────────────────────────
        # extract_faces returns a confidence score per face.
        # We skip faces below MIN_CONFIDENCE to ignore false detections.
        MIN_CONFIDENCE = 0.85

        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if not faces:
            return

        new_cache = {}

        for face in faces:
            # ── EDIT 6 applied: filter low-confidence detections ──────────────
            confidence = face.get("confidence", 1.0)
            if confidence < MIN_CONFIDENCE:
                continue

            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']

            # Skip tiny faces (noise/far-away detections)
            if w < 30 or h < 30:
                continue

            face_img = frame[y:y+h, x:x+w]
            cache_key = f"{x}_{y}_{w}_{h}"

            try:
                results = DeepFace.find(
                    img_path=face_img,
                    db_path=db_path,
                    # ── EDIT 5: Faster model ───────────────────────────────────
                    # Switched from VGG-Face to ArcFace.
                    # ArcFace is faster and has better accuracy.
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                if len(results) > 0 and len(results[0]) > 0:
                    # Extract person name from subfolder name, not filename
                    identity = os.path.basename(os.path.dirname(results[0]['identity'][0]))
                    new_cache[cache_key] = {
                        "label": identity,
                        "color": (0, 255, 0),  # green
                        "box": (x, y, w, h),
                        "timestamp": time.time(),
                        "frame": frame.copy(),
                        "known": True
                    }
                else:
                    new_cache[cache_key] = {
                        "label": "Unknown",
                        "color": (0, 0, 255),  # red
                        "box": (x, y, w, h),
                        "timestamp": time.time(),
                        "frame": frame.copy(),
                        "known": False
                    }

            except Exception as e:
                print(f"Recognition error: {e}")

        # ── EDIT 4: update shared cache safely using a lock ───────────────────
        with recognition_lock:
            face_cache.clear()
            face_cache.update(new_cache)

    except Exception as e:
        print(f"Detection error: {e}")
    finally:
        recognition_running = False


def custom_stream(db_path, model_name, detector_backend, source=0):
    """ Stream video, detect faces, and verify against the database. """
    global pending_frame, recognition_running

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise ValueError("Unable to open video source")

    frame_count = 0  # ── EDIT 1: counter for frame skipping

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        frame_count += 1

        # ── EDIT 1 applied: only queue a recognition job every FRAME_SKIP frames
        # and only if no recognition thread is already running ─────────────────
        if frame_count % FRAME_SKIP == 0 and not recognition_running:
            recognition_running = True
            t = threading.Thread(
                target=recognition_worker,
                args=(frame_resized.copy(), db_path, model_name, detector_backend),
                daemon=True
            )
            t.start()

        # ── EDIT 2 applied: draw cached results on every frame ────────────────
        now = time.time()
        with recognition_lock:
            for _, data in face_cache.items():
                # Skip stale cache entries
                if now - data["timestamp"] > CACHE_EXPIRY:
                    continue

                x, y, w, h = data["box"]
                color = data["color"]
                label = data["label"]

                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_resized, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ── EDIT 3 applied: throttled save ────────────────────────────
                folder = recognized_folder if data["known"] else unknown_folder
                save_frame(data["frame"], label, folder)

        cv2.imshow('Face Recognition', frame_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# Start the face recognition stream
custom_stream(
    db_path="C:/Users/AYA/Downloads/test",
    # ── EDIT 5 applied: switched model from VGG-Face → ArcFace ───────────────
    model_name="ArcFace",
    detector_backend="opencv",
    source=0,
)
