"""
Unified Eye, Face & Person Tracker for AI Interview.
Uses the exact UnifiedEyeFacePersonTracker logic - pass video path and get output.
Modify this file to improve tracking; the process_video() API stays the same.
"""

import cv2
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, 'solutions')
    if not MEDIAPIPE_AVAILABLE:
        print(f"❌ MediaPipe imported but missing 'solutions' attribute. Version: {getattr(mp, '__version__', 'unknown')}")
except (ImportError, AttributeError) as e:
    print(f"❌ MediaPipe import failed: {e}")
    mp = None
    MEDIAPIPE_AVAILABLE = False
import numpy as np
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import os
from typing import Optional, Dict, Any, List, Generator, Tuple

# FFmpeg path: env var > backend/FFMPEG/.../bin > backend/ffmpeg-8.0.1/bin > system PATH
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_FFMPEG_BIN = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
_CANDIDATE_PATHS = [
    os.getenv("FFMPEG_PATH"),
    str(_BACKEND_DIR / "FFMPEG" / "ffmpeg-master-latest-win64-gpl" / "ffmpeg-master-latest-win64-gpl" / "bin" / _FFMPEG_BIN),
    str(_BACKEND_DIR / "ffmpeg-8.0.1" / "bin" / _FFMPEG_BIN),
]
FFMPEG_PATH = "ffmpeg"
for p in _CANDIDATE_PATHS:
    if p and os.path.isfile(p):
        FFMPEG_PATH = p
        break

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# PyAV for WebM/VP9 support
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class UnifiedEyeFacePersonTracker:
    """Exact UnifiedEyeFacePersonTracker - modify this class for better output."""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE or mp is None:
            import subprocess
            try:
                mp_version = subprocess.check_output(['pip', 'show', 'mediapipe'], stderr=subprocess.DEVNULL).decode()
                print(f"MediaPipe package info:\n{mp_version}")
            except:
                pass
            raise RuntimeError(
                "MediaPipe is not installed correctly. "
                "Ensure Dockerfile includes: libgl1 libglib2.0-0 libgomp1 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 python3-dev. "
                "Rebuild Docker image after updating Dockerfile."
            )
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=10,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # YOLO Person Detection
        if YOLO_AVAILABLE:
            print("Loading YOLO model...")
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded")
        else:
            self.yolo_model = None

        # ========== EYE BLINK DETECTION ==========
        self.LEFT_EYE_BLINK = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_BLINK = [362, 385, 387, 263, 373, 380]
        self.EAR_THRESHOLD = 0.23

        # ========== EYE GAZE DETECTION ==========
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                         133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
                          263, 466, 388, 387, 386, 385, 384, 398]

        # ========== HEAD POSE DETECTION ==========
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        # ========== CALIBRATION ==========
        self.calibration_phase = 0
        self.calibration_labels = [
            "Face STRAIGHT + Eyes CENTER",
            "Face STRAIGHT + Eyes CENTER (confirm)",
            "Face STRAIGHT + Eyes EXTREME LEFT",
            "Face STRAIGHT + Eyes EXTREME RIGHT"
        ]

        self.neutral_yaw = 0
        self.neutral_pitch = 0
        self.face_samples = 0

        self.center_ratio = None
        self.left_ratio = None
        self.right_ratio = None

        self.calibrated = False

        # ========== SMOOTHING ==========
        self.smooth_yaw = 0
        self.smooth_pitch = 0
        self.alpha = 0.8

        self.ratio_buffer = []
        self.buffer_size = 5

        self.current_eye_direction = "CENTER"
        self.direction_hold_start = time.time()
        self.hold_time_required = 0.15

        # ========== RECORDING & LOGGING ==========
        self.video_writer = None
        self.log_file = None
        self.recording_start_time = None
        self.session_folder = None

        # ========== STATISTICS TRACKING ==========
        self.total_frames = 0
        self.eye_open_frames = 0
        self.eye_closed_frames = 0
        self.no_face_frames = 0
        self.multiple_faces_frames = 0

        # Track face count distribution
        self.face_count_distribution = {}

        # Track person count distribution
        self.person_count_distribution = {}
        self.no_person_frames = 0

        # Track head direction counts
        self.head_direction_counts = {
            "CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0,
            "TOP LEFT": 0, "TOP RIGHT": 0, "BOTTOM LEFT": 0, "BOTTOM RIGHT": 0,
            "---": 0
        }

        # Track eye gaze counts
        self.eye_gaze_counts = {
            "CENTER": 0, "LEFT": 0, "RIGHT": 0, "---": 0
        }

    def get_landmark_point(self, frame, landmark):
        h, w, _ = frame.shape
        return np.array([int(landmark.x * w), int(landmark.y * h)])

    def calculate_ear(self, frame, face_landmarks, eye_indices):
        """Calculate Eye Aspect Ratio"""
        points = [self.get_landmark_point(frame, face_landmarks.landmark[i])
                  for i in eye_indices]
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear, points

    def calculate_gaze_ratio(self, frame, face_landmarks):
        """Calculate eye gaze ratio"""
        left_iris_points = np.array([
            self.get_landmark_point(frame, face_landmarks.landmark[i])
            for i in self.LEFT_IRIS
        ])
        left_pupil = np.mean(left_iris_points, axis=0).astype(int)
        left_corner = self.get_landmark_point(frame, face_landmarks.landmark[33])
        right_corner = self.get_landmark_point(frame, face_landmarks.landmark[133])
        eye_width = right_corner[0] - left_corner[0]
        ratio = (left_pupil[0] - left_corner[0]) / eye_width if abs(eye_width) > 1e-6 else 0.5
        return ratio

    def smooth_gaze_ratio(self, ratio):
        """Smooth the gaze ratio"""
        self.ratio_buffer.append(ratio)
        if len(self.ratio_buffer) > self.buffer_size:
            self.ratio_buffer.pop(0)
        return np.mean(self.ratio_buffer)

    def detect_eye_direction(self, smooth_ratio):
        """Detect eye gaze direction"""
        if not self.calibrated:
            return "---"

        if self.center_ratio is None:
            return "---"
        left_r = self.left_ratio if self.left_ratio is not None else 0.2
        right_r = self.right_ratio if self.right_ratio is not None else 0.8

        if smooth_ratio < self.center_ratio:
            left_range = self.center_ratio - left_r
            left_range = max(left_range, 0.0001)
            normalized = (smooth_ratio - self.center_ratio) / left_range
        else:
            right_range = right_r - self.center_ratio
            right_range = max(right_range, 0.0001)
            normalized = (smooth_ratio - self.center_ratio) / right_range

        normalized = max(min(normalized, 1), -1)

        if normalized < -0.4:
            new_direction = "LEFT"
        elif normalized > 0.4:
            new_direction = "RIGHT"
        else:
            new_direction = "CENTER"

        if new_direction != self.current_eye_direction:
            if time.time() - self.direction_hold_start > self.hold_time_required:
                self.current_eye_direction = new_direction
                self.direction_hold_start = time.time()
        else:
            self.direction_hold_start = time.time()

        return self.current_eye_direction

    def calculate_head_pose(self, frame, face_landmarks):
        """Calculate head pose"""
        h, w, _ = frame.shape
        mesh_points = np.array([
            (int(p.x * w), int(p.y * h))
            for p in face_landmarks.landmark
        ])

        image_points = np.array([
            mesh_points[1], mesh_points[152], mesh_points[263],
            mesh_points[33], mesh_points[287], mesh_points[57]
        ], dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, cam_matrix, dist_coeffs
        )

        if not success:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1]

    def detect_head_direction(self, pitch, yaw):
        """Detect head direction (left/right corrected so display matches actual head turn)."""
        t = 7
        if yaw > t and pitch < -t:
            return "TOP LEFT"
        elif yaw < -t and pitch < -t:
            return "TOP RIGHT"
        elif yaw > t and pitch > t:
            return "BOTTOM LEFT"
        elif yaw < -t and pitch > t:
            return "BOTTOM RIGHT"
        elif yaw > t:
            return "LEFT"
        elif yaw < -t:
            return "RIGHT"
        elif pitch < -t:
            return "UP"
        elif pitch > t:
            return "DOWN"
        else:
            return "CENTER"

    def draw_eye_visuals(self, frame, face_landmarks):
        """Draw iris and eye outlines"""
        h, w, _ = frame.shape
        try:
            left_iris_points = np.array([
                self.get_landmark_point(frame, face_landmarks.landmark[i])
                for i in self.LEFT_IRIS
            ])
            left_pupil = np.mean(left_iris_points, axis=0).astype(int)
            left_radius = max(int(np.linalg.norm(left_iris_points[0] - left_iris_points[2]) / 2), 2)
            cv2.circle(frame, tuple(left_pupil), left_radius, (0, 255, 0), 2)
            cv2.circle(frame, tuple(left_pupil), 4, (0, 255, 255), -1)

            right_iris_points = np.array([
                self.get_landmark_point(frame, face_landmarks.landmark[i])
                for i in self.RIGHT_IRIS
            ])
            right_pupil = np.mean(right_iris_points, axis=0).astype(int)
            right_radius = max(int(np.linalg.norm(right_iris_points[0] - right_iris_points[2]) / 2), 2)
            cv2.circle(frame, tuple(right_pupil), right_radius, (0, 255, 0), 2)
            cv2.circle(frame, tuple(right_pupil), 4, (0, 255, 255), -1)

            left_eye_points = np.array([
                self.get_landmark_point(frame, face_landmarks.landmark[i])
                for i in self.LEFT_EYE
            ], dtype=np.int32)
            right_eye_points = np.array([
                self.get_landmark_point(frame, face_landmarks.landmark[i])
                for i in self.RIGHT_EYE
            ], dtype=np.int32)
            cv2.polylines(frame, [left_eye_points], True, (255, 0, 255), 1)
            cv2.polylines(frame, [right_eye_points], True, (255, 0, 255), 1)
        except Exception:
            pass

    def log_tracking_data(self, person_count, face_count, eye_status, head_direction, eye_direction):
        """Write tracking data to stats (no file in batch mode)"""
        if self.recording_start_time:
            self.total_frames += 1

            if eye_status == "EYES OPEN":
                self.eye_open_frames += 1
            elif eye_status == "EYES CLOSED":
                self.eye_closed_frames += 1
            elif eye_status == "NO FACE":
                self.no_face_frames += 1
            elif eye_status == "MULTIPLE FACES":
                self.multiple_faces_frames += 1

            self.face_count_distribution[face_count] = self.face_count_distribution.get(face_count, 0) + 1
            self.person_count_distribution[person_count] = self.person_count_distribution.get(person_count, 0) + 1
            if person_count == 0:
                self.no_person_frames += 1

            if head_direction in self.head_direction_counts:
                self.head_direction_counts[head_direction] += 1
            if eye_direction in self.eye_gaze_counts:
                self.eye_gaze_counts[eye_direction] += 1

    def process_frame(self, frame, fps: float, frame_idx: int, frame_log: List[Dict]):
        """Main processing - same logic as original."""
        h, w, _ = frame.shape

        # ========== YOLO PERSON DETECTION ==========
        person_count = 0
        if self.yolo_model:
            try:
                yolo_results = self.yolo_model(frame, classes=[0], verbose=False)
                person_count = len(yolo_results[0].boxes)
                frame = yolo_results[0].plot()
            except Exception:
                pass

        # ========== MEDIAPIPE FACE DETECTION ==========
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        eye_status = "NO FACE"
        head_direction = "---"
        eye_direction = "---"
        avg_ear = 0.0
        face_count = 0

        if not results.multi_face_landmarks:
            eye_status = "NO FACE"
            face_count = 0
            face_count_msg = "NO FACE DETECTED"
            face_color = (0, 0, 255)
        elif len(results.multi_face_landmarks) > 1:
            face_count = len(results.multi_face_landmarks)
            eye_status = "MULTIPLE FACES"
            face_count_msg = f"MULTIPLE FACES DETECTED ({face_count} faces)"
            face_color = (0, 165, 255)
        else:
            face_count = 1
            face_count_msg = None
            face_landmarks = results.multi_face_landmarks[0]

            left_ear, left_points = self.calculate_ear(frame, face_landmarks, self.LEFT_EYE_BLINK)
            right_ear, right_points = self.calculate_ear(frame, face_landmarks, self.RIGHT_EYE_BLINK)
            avg_ear = (left_ear + right_ear) / 2.0

            cv2.polylines(frame, [np.array(left_points)], True, (128, 128, 128), 1)
            cv2.polylines(frame, [np.array(right_points)], True, (128, 128, 128), 1)

            if avg_ear < self.EAR_THRESHOLD:
                eye_status = "EYES CLOSED"
                eye_color = (0, 0, 255)
            else:
                eye_status = "EYES OPEN"
                eye_color = (0, 255, 0)

                if self.calibrated:
                    self.draw_eye_visuals(frame, face_landmarks)

                    pitch, yaw = self.calculate_head_pose(frame, face_landmarks)
                    pitch -= self.neutral_pitch
                    yaw -= self.neutral_yaw
                    yaw = -yaw

                    self.smooth_pitch = self.alpha * self.smooth_pitch + (1 - self.alpha) * pitch
                    self.smooth_yaw = self.alpha * self.smooth_yaw + (1 - self.alpha) * yaw

                    head_direction = self.detect_head_direction(self.smooth_pitch, self.smooth_yaw)

                    ratio = self.calculate_gaze_ratio(frame, face_landmarks)
                    smooth_ratio = self.smooth_gaze_ratio(ratio)
                    eye_direction = self.detect_eye_direction(smooth_ratio)

        # Log to frame_log for full_log text
        secs = (frame_idx - 1) / fps if frame_idx > 0 else 0
        time_str = f"{int(secs // 60):02d}:{secs % 60:06.3f}"
        frame_log.append({
            "time": time_str,
            "persons": person_count,
            "faces": face_count,
            "eye_status": eye_status,
            "head_dir": head_direction,
            "eye_gaze": eye_direction,
        })

        if self.calibrated:
            self.log_tracking_data(person_count, face_count, eye_status, head_direction, eye_direction)

        # Overlay
        y_offset = 40
        cv2.putText(frame, f"Persons: {person_count}",
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 40

        if face_count == 0:
            cv2.putText(frame, "NO FACE DETECTED",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif face_count > 1:
            cv2.putText(frame, f"MULTIPLE FACES ({face_count})",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"Eye Status: {eye_status}",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if eye_status == "EYES OPEN" else (0, 0, 255), 2)
            y_offset += 40
            cv2.putText(frame, f"Head Direction: {head_direction}",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40
            cv2.putText(frame, f"Eye Gaze: {eye_direction}",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def calibrate_from_images(self, image_paths: List[str]) -> bool:
        """Calibrate from 4 pre-captured images (same logic as capture_calibration)."""
        if not image_paths or len(image_paths) != 4:
            return False
        ratio_buffer = []
        for phase, path in enumerate(image_paths):
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
                continue
            lm = results.multi_face_landmarks[0]
            h, w = img.shape[:2]
            pitch, yaw = self.calculate_head_pose(img, lm)
            ratio = self.calculate_gaze_ratio(img, lm)
            ratio_buffer.append(ratio)
            if len(ratio_buffer) > self.buffer_size:
                ratio_buffer.pop(0)
            smooth_ratio = np.mean(ratio_buffer) if ratio_buffer else ratio

            if phase == 0:
                self.neutral_yaw = yaw
                self.neutral_pitch = pitch
                self.center_ratio = smooth_ratio
            elif phase == 1:
                self.center_ratio = np.mean(ratio_buffer) if ratio_buffer else self.center_ratio
            elif phase == 2:
                self.left_ratio = np.mean(ratio_buffer) if ratio_buffer else 0.2
            elif phase == 3:
                self.right_ratio = np.mean(ratio_buffer) if ratio_buffer else 0.8
                self.calibrated = True
        return self.calibrated


def _iter_frames_pyav(video_path: str) -> Generator[Tuple[np.ndarray, int, float, int, int], None, None]:
    """Iterate video frames using PyAV (WebM/VP9)."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        rate = getattr(stream, "average_rate", None) or getattr(stream, "frame_rate", None) or getattr(stream, "framerate", None)
        fps = float(rate) if rate else 30.0
        if fps <= 0:
            fps = 30.0
        idx = 0
        for frame in container.decode(video=0):
            try:
                arr = frame.to_ndarray(format="bgr24")
            except Exception:
                try:
                    arr = frame.reformat(format="bgr24").to_ndarray()
                except Exception:
                    continue
            h, w = arr.shape[:2]
            idx += 1
            yield arr, idx, fps, w, h
    finally:
        container.close()


def _iter_frames_opencv(video_path: str) -> Generator[Tuple[np.ndarray, int, float, int, int], None, None]:
    """Iterate video frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = max(cap.get(cv2.CAP_PROP_FPS) or 30, 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        idx += 1
        h, w = frame.shape[:2]
        yield frame, idx, fps, w, h
    cap.release()


def _build_full_log(
    frame_log: List[Dict],
    duration_sec: float,
    tracker: UnifiedEyeFacePersonTracker,
) -> str:
    """Build full log text in UnifiedEyeFacePersonTracker format."""
    total = len(frame_log)
    if total == 0:
        total = 1
    lines = []
    sep = "=" * 100
    hsep = "-" * 100

    lines.append(sep)
    lines.append("UNIFIED EYE, FACE & PERSON TRACKING LOG")
    lines.append(f"Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
    lines.append(sep)
    lines.append(f"{'Time':<15} | {'Persons':<8} | {'Faces':<8} | {'Eye Status':<15} | {'Head Dir':<15} | {'Eye Gaze':<15}")
    lines.append(hsep)

    for row in frame_log:
        t = str(row.get("time", ""))
        p = str(row.get("persons", ""))
        f = str(row.get("faces", ""))
        eye = str(row.get("eye_status", "---"))
        head = str(row.get("head_dir", "---"))
        gaze = str(row.get("eye_gaze", "---"))
        lines.append(f"{t:<15} | {p:<8} | {f:<8} | {eye:<15} | {head:<15} | {gaze:<15}")

    lines.append(hsep)
    lines.append("")
    lines.append(sep)
    lines.append("SUMMARY STATISTICS")
    lines.append(sep)
    lines.append("")

    min_part = int(duration_sec // 60)
    sec_part = duration_sec % 60
    lines.append(f"Total Recording Duration: {min_part:02d}:{sec_part:06.3f} ({duration_sec:.3f} seconds)")
    lines.append(f"Total Frames Logged: {total}")
    lines.append("")

    # Person stats
    lines.append(hsep)
    lines.append("PERSON DETECTION STATISTICS (YOLO)")
    lines.append(hsep)
    for k in sorted(tracker.person_count_distribution.keys()):
        v = tracker.person_count_distribution[k]
        pct = (v / total) * 100
        dur = (v / total) * duration_sec
        lines.append(f"{k:<15} | {v:<10} | {pct:>10.2f}% | {dur:>13.3f}")
    no_person = tracker.person_count_distribution.get(0, 0)
    no_person_pct = (no_person / total) * 100
    no_person_sec = (no_person / total) * duration_sec
    lines.append(f"\nNo Person Detected: {no_person_pct:6.2f}% ({no_person_sec:8.3f} seconds)\n")

    # Face stats
    lines.append(hsep)
    lines.append("FACE DETECTION STATISTICS (MediaPipe)")
    lines.append(hsep)
    for k in sorted(tracker.face_count_distribution.keys()):
        v = tracker.face_count_distribution[k]
        pct = (v / total) * 100
        dur = (v / total) * duration_sec
        lines.append(f"{k:<15} | {v:<10} | {pct:>10.2f}% | {dur:>13.3f}")
    single_face_frames = tracker.eye_open_frames + tracker.eye_closed_frames
    single_face_pct = (single_face_frames / total) * 100
    single_face_sec = (single_face_frames / total) * duration_sec
    no_face_pct = (tracker.no_face_frames / total) * 100
    no_face_sec = (tracker.no_face_frames / total) * duration_sec
    multi_face_pct = (tracker.multiple_faces_frames / total) * 100
    multi_face_sec = (tracker.multiple_faces_frames / total) * duration_sec
    lines.append(f"\nNo Face Detected:       {no_face_pct:6.2f}% ({no_face_sec:8.3f} seconds)")
    lines.append(f"Single Face (Tracked):  {single_face_pct:6.2f}% ({single_face_sec:8.3f} seconds)")
    lines.append(f"Multiple Faces:         {multi_face_pct:6.2f}% ({multi_face_sec:8.3f} seconds)\n")

    # Eye status
    lines.append(hsep)
    lines.append("EYE STATUS STATISTICS (Single Face Only)")
    lines.append(hsep)
    eo_pct = (tracker.eye_open_frames / total) * 100
    eo_sec = (tracker.eye_open_frames / total) * duration_sec
    ec_pct = (tracker.eye_closed_frames / total) * 100
    ec_sec = (tracker.eye_closed_frames / total) * duration_sec
    lines.append(f"Eyes Open:    {eo_pct:6.2f}% ({eo_sec:8.3f} seconds)")
    lines.append(f"Eyes Closed:  {ec_pct:6.2f}% ({ec_sec:8.3f} seconds)\n")

    # Head direction
    lines.append(hsep)
    lines.append("HEAD DIRECTION STATISTICS (Single Face Only)")
    lines.append(hsep)
    for d in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN",
              "TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT", "---"]:
        c = tracker.head_direction_counts.get(d, 0)
        if c > 0:
            pct = (c / total) * 100
            sec = (c / total) * duration_sec
            lines.append(f"{d:<15}: {pct:6.2f}% ({sec:8.3f} seconds)")
    lines.append("")

    # Eye gaze
    lines.append(hsep)
    lines.append("EYE GAZE STATISTICS (Single Face Only)")
    lines.append(hsep)
    for g in ["CENTER", "LEFT", "RIGHT", "---"]:
        c = tracker.eye_gaze_counts.get(g, 0)
        if c > 0:
            pct = (c / total) * 100
            sec = (c / total) * duration_sec
            lines.append(f"{g:<15}: {pct:6.2f}% ({sec:8.3f} seconds)")

    lines.append("")
    lines.append(sep)
    lines.append(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
    lines.append(sep)
    return "\n".join(lines)


def process_video(
    video_path: str,
    calibration_image_paths: Optional[List[str]] = None,
    calibration_image_urls: Optional[List[str]] = None,
    download_fn=None,
    produce_annotated_video: bool = True,
) -> Dict[str, Any]:
    """
    Process video with UnifiedEyeFacePersonTracker.
    Pass video path and optionally calibration images. Returns metrics and annotated video path.

    Args:
        video_path: Path to video file (WebM or MP4)
        calibration_image_paths: Local paths to 4 calibration images
        calibration_image_urls: S3 URLs of 4 calibration images (requires download_fn)
        download_fn: Sync function(url) -> bytes for downloading
        produce_annotated_video: If True, write annotated video with overlays

    Returns:
        Dict with score, full_log, annotated_video_path, eye_open_pct, head_center_pct, etc.
    """
    tracker = UnifiedEyeFacePersonTracker()

    # Calibrate from images
    paths = list(calibration_image_paths) if calibration_image_paths else []
    temp_paths = []
    if not paths and calibration_image_urls and download_fn:
        for url in (calibration_image_urls or [])[:4]:
            try:
                data = download_fn(url)
                if data:
                    fd, p = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    with open(p, "wb") as f:
                        f.write(data)
                    paths.append(p)
                    temp_paths.append(p)
            except Exception:
                pass
    if paths:
        tracker.calibrate_from_images(paths)
        for p in temp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

    if not tracker.calibrated:
        # Use defaults
        tracker.center_ratio = 0.5
        tracker.left_ratio = 0.2
        tracker.right_ratio = 0.8
        tracker.calibrated = True

    use_pyav = video_path.lower().endswith(".webm") and PYAV_AVAILABLE
    if use_pyav:
        frame_iter = _iter_frames_pyav(video_path)
    else:
        frame_iter = _iter_frames_opencv(video_path)

    try:
        first_frame, first_idx, fps, w, h = next(frame_iter)
    except (StopIteration, Exception) as e:
        return {"error": str(e), "score": 0}

    tracker.recording_start_time = time.time()
    frame_log: List[Dict] = []

    # Output video
    annotated_path: Optional[str] = None
    video_writer = None

    if produce_annotated_video:
        fd, annotated_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        # Use OpenCV VideoWriter (PyAV libx264 has compatibility issues on some platforms)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))

    def write_frame(annotated: np.ndarray):
        if video_writer:
            video_writer.write(annotated)

    count = 0
    try:
        annotated = tracker.process_frame(first_frame, fps, first_idx, frame_log)
        if produce_annotated_video:
            write_frame(annotated)
        count += 1

        for frame, idx, _, _, _ in frame_iter:
            annotated = tracker.process_frame(frame, fps, idx, frame_log)
            if produce_annotated_video:
                write_frame(annotated)
            count += 1
            if count % 100 == 0:
                print(f"[UnifiedTracker] Processed {count} frames...")
    finally:
        if video_writer:
            video_writer.release()

    # Convert mp4v to H.264 for browser playback (mp4v not supported in browsers)
    if produce_annotated_video and annotated_path and os.path.exists(annotated_path):
        h264_path = annotated_path.replace(".mp4", "_h264.mp4")
        try:
            _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            result = subprocess.run(
                [
                    FFMPEG_PATH, "-y", "-i", annotated_path,
                    "-c:v", "libx264", "-preset", "fast",
                    "-pix_fmt", "yuv420p", h264_path
                ],
                capture_output=True,
                timeout=300,
                creationflags=_no_window,
            )
            if result.returncode == 0 and os.path.exists(h264_path):
                try:
                    os.unlink(annotated_path)
                except Exception:
                    pass
                annotated_path = h264_path
                print("[UnifiedTracker] Converted to H.264 for browser playback")
            else:
                err = (result.stderr or b"").decode("utf-8", errors="ignore")[:500]
                print(f"[UnifiedTracker] FFmpeg conversion failed: {err}")
        except subprocess.TimeoutExpired:
            print("[UnifiedTracker] FFmpeg conversion timed out, keeping mp4v")
        except FileNotFoundError:
            print("[UnifiedTracker] FFmpeg not found, keeping mp4v (video may not play in browser)")
        except Exception as e:
            print(f"[UnifiedTracker] FFmpeg error: {e}, keeping mp4v")

    tracker.face_mesh.close()

    total_frames = tracker.total_frames or count
    duration_sec = total_frames / fps if fps > 0 else 0
    single_face_frames = tracker.eye_open_frames + tracker.eye_closed_frames

    eye_open_pct = (tracker.eye_open_frames / single_face_frames * 100) if single_face_frames > 0 else 0
    head_center_pct = (tracker.head_direction_counts.get("CENTER", 0) / single_face_frames * 100) if single_face_frames > 0 else 0
    eye_center_pct = (tracker.eye_gaze_counts.get("CENTER", 0) / single_face_frames * 100) if single_face_frames > 0 else 0
    single_face_pct = (single_face_frames / total_frames * 100) if total_frames > 0 else 0
    no_face_pct = (tracker.no_face_frames / total_frames * 100) if total_frames > 0 else 0
    multiple_faces_pct = (tracker.multiple_faces_frames / total_frames * 100) if total_frames > 0 else 0

    score = (
        single_face_pct * 0.25 +
        min(eye_open_pct, 100) * 0.35 +
        min(head_center_pct, 100) * 0.25 +
        min(eye_center_pct, 100) * 0.15 -
        no_face_pct * 0.2 -
        multiple_faces_pct * 0.3
    )
    score = max(0, min(100, round(score)))

    full_log = _build_full_log(frame_log, duration_sec, tracker)

    eye_closed_pct = (tracker.eye_closed_frames / single_face_frames * 100) if single_face_frames > 0 else 0
    frame_to_sec = (1.0 / fps) if fps > 0 else 0
    person_total = sum(tracker.person_count_distribution.values()) or 1
    face_total = sum(tracker.face_count_distribution.values()) or 1
    person_stats = [
        {"count": str(k), "frames": v, "pct": round(v / person_total * 100, 2), "duration_sec": round(v * frame_to_sec, 3)}
        for k, v in sorted(tracker.person_count_distribution.items())
    ]
    face_stats = [
        {"count": str(k), "frames": v, "pct": round(v / face_total * 100, 2), "duration_sec": round(v * frame_to_sec, 3)}
        for k, v in sorted(tracker.face_count_distribution.items())
    ]
    eye_status_stats = {
        "eyes_open_pct": round(eye_open_pct, 2),
        "eyes_open_sec": round(tracker.eye_open_frames * frame_to_sec, 3),
        "eyes_closed_pct": round(eye_closed_pct, 2),
        "eyes_closed_sec": round(tracker.eye_closed_frames * frame_to_sec, 3),
    }
    sf_total = single_face_frames or 1
    head_direction_stats = [
        {"dir": d, "pct": round(tracker.head_direction_counts.get(d, 0) / sf_total * 100, 2),
         "duration_sec": round(tracker.head_direction_counts.get(d, 0) * frame_to_sec, 3)}
        for d in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT", "---"]
        if tracker.head_direction_counts.get(d, 0) > 0
    ]
    eye_gaze_stats = [
        {"dir": d, "pct": round(tracker.eye_gaze_counts.get(d, 0) / sf_total * 100, 2),
         "duration_sec": round(tracker.eye_gaze_counts.get(d, 0) * frame_to_sec, 3)}
        for d in ["CENTER", "LEFT", "RIGHT", "---"]
        if tracker.eye_gaze_counts.get(d, 0) > 0
    ]

    def _get_suggestion(s, eo, hc, nf, ec):
        if s >= 80:
            return "Strong eye contact and engagement throughout the interview."
        if s >= 60:
            return "Good overall engagement. Consider maintaining more consistent eye contact."
        if nf > 20:
            return "Face was frequently not visible. Ensure camera remains focused on your face."
        if eo < 70:
            return "Low eye openness detected. Ensure adequate lighting and camera positioning."
        if hc < 50:
            return "Head was often turned away. Try to face the camera more consistently."
        if ec < 50:
            return "Eye gaze was often directed away from center. Maintain eye contact with the camera."
        return "Review the recording to improve engagement and eye contact."

    suggestion = _get_suggestion(score, eye_open_pct, head_center_pct, no_face_pct, eye_center_pct)

    return {
        "score": score,
        "calibrated": len(paths) >= 4,
        "eye_open_pct": round(eye_open_pct, 1),
        "eye_closed_pct": round(eye_closed_pct, 1),
        "head_center_pct": round(head_center_pct, 1),
        "eye_gaze_center_pct": round(eye_center_pct, 1),
        "single_face_pct": round(single_face_pct, 1),
        "no_face_pct": round(no_face_pct, 1),
        "multiple_faces_pct": round(multiple_faces_pct, 1),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "frame_log": frame_log,
        "full_log": full_log,
        "annotated_video_path": annotated_path,
        "summary_stats": {
            "duration_sec": round(duration_sec, 3),
            "total_frames_logged": len(frame_log),
            "person_stats": person_stats,
            "face_stats": face_stats,
            "eye_status_stats": eye_status_stats,
            "head_direction_stats": head_direction_stats,
            "eye_gaze_stats": eye_gaze_stats,
            "no_face_pct": round(no_face_pct, 2),
            "no_face_sec": round(tracker.no_face_frames * frame_to_sec, 3),
            "single_face_pct": round(single_face_pct, 2),
            "single_face_sec": round(single_face_frames * frame_to_sec, 3),
            "multiple_faces_pct": round(multiple_faces_pct, 2),
            "multiple_faces_sec": round(tracker.multiple_faces_frames * frame_to_sec, 3),
        },
        "head_direction_distribution": dict(tracker.head_direction_counts),
        "eye_gaze_distribution": dict(tracker.eye_gaze_counts),
        "person_count_distribution": {str(k): v for k, v in tracker.person_count_distribution.items()},
        "face_count_distribution": {str(k): v for k, v in tracker.face_count_distribution.items()},
        "suggestion": suggestion,
    }
