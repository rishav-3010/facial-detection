import cv2
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import pickle
from collections import deque

class LivenessDetector:
    """Detect if face is real or a photo/screen"""
    
    def __init__(self, history_size=10):
        self.motion_history = deque(maxlen=history_size)
        self.prev_gray = None
        self.blink_counter = 0
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def detect_motion(self, face_roi):
        """Detect micro-movements in face region"""
        # Convert to grayscale and resize to a fixed size for stable optical flow
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        target_size = (128, 128)
        try:
            gray_resized = cv2.resize(gray, target_size)
        except Exception:
            # If resizing fails, skip motion detection for this frame
            self.prev_gray = None
            return None

        # If we don't have a previous frame yet, store and wait
        if self.prev_gray is None:
            self.prev_gray = gray_resized.copy()
            return None

        # Ensure shapes match; if not, reset previous and skip flow to avoid assertion
        if self.prev_gray.shape != gray_resized.shape:
            self.prev_gray = gray_resized.copy()
            self.motion_history.clear()
            return None

        # Calculate optical flow between fixed-size frames
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_resized, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = np.mean(magnitude)
        self.motion_history.append(motion_score)

        # update previous frame
        self.prev_gray = gray_resized.copy()

        # Real faces have continuous small movements (0.5-5.0)
        # Photos/screens have very little motion (<0.3)
        if len(self.motion_history) >= 5:
            avg_motion = np.mean(self.motion_history)
            return avg_motion > 0.5
        return None  # Not enough data yet
    
    def detect_texture(self, face_roi):
        """Analyze texture patterns - screens/photos have different texture"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness/texture measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate color variance in small patches
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        patches = []
        h, w = hsv.shape[:2]
        patch_size = 20
        
        for i in range(0, h-patch_size, patch_size):
            for j in range(0, w-patch_size, patch_size):
                patch = hsv[i:i+patch_size, j:j+patch_size]
                patches.append(np.std(patch))
        
        texture_variance = np.mean(patches)
        
        # Real faces: laplacian_var > 100, texture_variance > 15
        # Photos/screens: lower values due to compression/pixelation
        is_real = laplacian_var > 100 and texture_variance > 15
        return is_real
    
    def detect_screen_glare(self, face_roi):
        """Detect screen glare patterns"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Find very bright spots (typical of screen reflections)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_percentage = np.count_nonzero(bright_mask) / bright_mask.size
        
        # Screens often have uniform bright spots (>5% very bright pixels)
        return bright_percentage < 0.05
    
    def check_liveness(self, face_roi):
        """Combined liveness check"""
        checks = {
            'motion': self.detect_motion(face_roi),
            'texture': self.detect_texture(face_roi),
            'glare': self.detect_screen_glare(face_roi)
        }
        
        # Need at least 2 out of 3 checks to pass
        # Motion can be None initially, so handle that
        valid_checks = [v for v in checks.values() if v is not None]
        
        if len(valid_checks) < 2:
            return None, checks  # Not enough data
        
        passed = sum(valid_checks)
        is_live = passed >= 2
        
        return is_live, checks

def preprocess_face(frame, face_cascade, size=(256, 256)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    processed_faces = []

    for (x, y, w, h) in faces:
        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        face_img = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_img, size)
        processed_faces.append((face_resized, (x1, y1, x2, y2), face_img))
    return processed_faces

def list_available_cameras():
    """List all available camera devices"""
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    # List available cameras
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("[ERROR] No cameras found!")
        return
    
    print("\nAvailable cameras:")
    for idx in available_cameras:
        print(f"Camera {idx}")
    
    camera_idx = int(input(f"\nSelect camera index (0-{len(available_cameras)-1}): "))
    if camera_idx not in available_cameras:
        print("[ERROR] Invalid camera index!")
        return

    # Load models and embeddings
    try:
        print("[INFO] Loading models and embeddings...")
        embedding_model = load_model("embedding_model.h5", compile=False)
        with open("embeddings.pkl", "rb") as f:
            emb_dict = pickle.load(f)
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print("[ERROR] Could not load models:", e)
        return

    # Initialize face detection and liveness detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    liveness_detector = LivenessDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_idx}")
        return

    print("[INFO] Starting camera feed...")
    print("[INFO] Liveness detection active - photos/screens will be rejected")
    print("[INFO] Press 'q' to quit")

    # Set up face recognition
    embeddings = emb_dict['embeddings']
    labels = emb_dict['labels']
    label_map = emb_dict['label_map']

    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't receive frame from camera")
            break

        faces = preprocess_face(frame, face_cascade)
        
        for face_img, (x1, y1, x2, y2), face_roi in faces:
            # Check liveness first
            is_live, checks = liveness_detector.check_liveness(face_roi)
            
            if is_live is None:
                # Still collecting data
                color = (255, 165, 0)  # Orange
                text = "Analyzing..."
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                continue
            
            if not is_live:
                # Detected as photo/screen
                color = (0, 0, 255)  # Red
                text = "FAKE - Photo/Screen Detected"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show debug info
                y_offset = y2 + 20
                cv2.putText(frame, f"Motion: {checks['motion']}", 
                           (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Texture: {checks['texture']}", 
                           (x1, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                continue
            
            # If liveness passed, do face recognition
            img_array = np.expand_dims(preprocess_input(face_img), axis=0)
            emb = embedding_model.predict(img_array, verbose=0)

            if len(emb.shape) > 2:
                emb = emb.reshape(emb.shape[0], -1)

            dist, idx = nbrs.kneighbors(emb)
            label = label_map[labels[idx[0][0]]]

            if dist[0][0] < 0.6:
                color = (0, 255, 0)  # Green
                text = f"LIVE - Valid ({label})"
            else:
                color = (255, 255, 0)  # Yellow
                text = "LIVE - Unknown Person"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show liveness status
            cv2.putText(frame, "Liveness: PASS", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Face Recognition with Liveness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()