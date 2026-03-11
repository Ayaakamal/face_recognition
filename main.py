# Real-time Face Recognition using DeepFace + OpenCV

from deepface import DeepFace  # face detection & recognition
import cv2                     # webcam capture & drawing
import os                      # file paths & folder creation
import threading               # run recognition in background
import time                    # cache expiry timing
from datetime import datetime  # timestamp for saved images


# ── Output folders (outside project to keep repo clean) ───────────────────────
OUTPUT_DIR        = os.path.join(os.path.expanduser("~"), "face_reco_output")
recognized_folder = os.path.join(OUTPUT_DIR, "recognized")
unknown_folder    = os.path.join(OUTPUT_DIR, "unknown")
os.makedirs(recognized_folder, exist_ok=True)
os.makedirs(unknown_folder, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
FRAME_SKIP   = 4    # process 1 in every 4 frames to reduce CPU load
CACHE_EXPIRY = 3.0  # seconds before a result is considered stale

# ── Shared state ───────────────────────────────────────────────────────────────
face_cache          = {}              # latest recognition results per face
already_saved       = set()          # tracks who was already saved this session
recognition_lock    = threading.Lock()
pending_frame       = None
recognition_running = False


def save_frame(frame, identity, folder):
    """Save one snapshot per person per session."""
    if identity in already_saved:
        return
    already_saved.add(identity)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{folder}/{identity}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")


def recognition_worker(frame, db_path, model_name, detector_backend):
    """Background thread: detect faces → match against database → update cache."""
    global recognition_running
    try:
        MIN_CONFIDENCE = 0.85  # ignore weak detections

        # Detect all faces in the frame
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        if not faces:
            return

        new_cache = {}

        for face in faces:
            # Skip low-confidence or tiny faces
            if face.get("confidence", 1.0) < MIN_CONFIDENCE:
                continue

            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']

            if w < 30 or h < 30:  # too small = noise
                continue

            face_img  = frame[y:y+h, x:x+w]  # crop face region
            cache_key = f"{x}_{y}_{w}_{h}"

            try:
                # Search the face database for a match
                results = DeepFace.find(
                    img_path=face_img,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                if len(results) > 0 and len(results[0]) > 0:
                    # Get person name from subfolder: database/AYA/photo.jpg → "AYA"
                    identity = os.path.basename(os.path.dirname(results[0]['identity'][0]))
                    new_cache[cache_key] = {
                        "label": identity, "color": (0, 255, 0),  # green
                        "box": (x, y, w, h), "timestamp": time.time(),
                        "frame": frame.copy(), "known": True
                    }
                else:
                    new_cache[cache_key] = {
                        "label": "Unknown", "color": (0, 0, 255),  # red
                        "box": (x, y, w, h), "timestamp": time.time(),
                        "frame": frame.copy(), "known": False
                    }

            except Exception as e:
                print(f"Recognition error: {e}")

        # Push results to shared cache safely
        with recognition_lock:
            face_cache.clear()
            face_cache.update(new_cache)

    except Exception as e:
        print(f"Detection error: {e}")
    finally:
        recognition_running = False  # allow next thread to start


def custom_stream(db_path, model_name, detector_backend, source=0):
    """Main loop: read webcam → run recognition → draw results → save snapshots."""
    global pending_frame, recognition_running

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow = stable on Windows
    if not cap.isOpened():
        raise ValueError("Could not open webcam.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        frame_count  += 1

        # Launch recognition thread every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0 and not recognition_running:
            recognition_running = True
            threading.Thread(
                target=recognition_worker,
                args=(frame_resized.copy(), db_path, model_name, detector_backend),
                daemon=True
            ).start()

        # Draw cached results on every frame
        now = time.time()
        with recognition_lock:
            for _, data in face_cache.items():
                if now - data["timestamp"] > CACHE_EXPIRY:
                    continue  # result too old, skip

                x, y, w, h = data["box"]
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), data["color"], 2)
                cv2.putText(frame_resized, data["label"], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, data["color"], 2)

                folder = recognized_folder if data["known"] else unknown_folder
                save_frame(data["frame"], data["label"], folder)

        cv2.imshow('Face Recognition', frame_resized)

        # Q, q, or ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Run ────────────────────────────────────────────────────────────────────────
custom_stream(
    db_path          = "C:/Users/AYA/Downloads/test",  # folder with subfolders per person
    model_name       = "ArcFace",    # best speed/accuracy balance
    detector_backend = "opencv",     # fast & stable on Windows
    source           = 0,            # 0 = default webcam
)
