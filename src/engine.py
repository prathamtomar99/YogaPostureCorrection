import mediapipe as mp
import numpy as np
import cv2
import os
import urllib.request

class AITrackingEngine:
    def __init__(self, max_people=1):
        self.model_path = 'pose_landmarker_lite.task'
        self._download_model_if_needed()

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max_people 
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe model (this only happens once)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)

    def get_person_count_and_landmarks(self, mp_image, timestamp_ms):
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.pose_landmarks:
            return 0, []
        
        person_count = len(result.pose_landmarks)
        return person_count, result.pose_landmarks[0]

class YogaAnalyzer:
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (27, 31), (28, 32)
    ]

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates the angle at point b given coordinates a, b, c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return angle if angle <= 180 else 360 - angle

    @staticmethod
    def _xy(landmarks, idx):
        return [landmarks[idx].x, landmarks[idx].y]

    @staticmethod
    def _avg(a, b):
        return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

    @staticmethod
    def _is_visible(landmarks, idx, min_visibility=0.4):
        if not hasattr(landmarks[idx], 'visibility'):
            return True
        return landmarks[idx].visibility >= min_visibility

    @staticmethod
    def draw_landmarks_and_skeleton(frame, landmarks, min_visibility=0.35, bad_joints=None):
        h, w, _ = frame.shape
        if bad_joints is None:
            bad_joints = set()

        for i, j in YogaAnalyzer.POSE_CONNECTIONS:
            if not (YogaAnalyzer._is_visible(landmarks, i, min_visibility) and YogaAnalyzer._is_visible(landmarks, j, min_visibility)):
                continue
            p1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
            p2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
            bone_color = (0, 80, 255) if (i in bad_joints or j in bad_joints) else (255, 200, 0)
            cv2.line(frame, p1, p2, bone_color, 2)

        for idx, lm in enumerate(landmarks):
            if not YogaAnalyzer._is_visible(landmarks, idx, min_visibility):
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            dot_color = (0, 0, 255) if idx in bad_joints else (0, 255, 255)
            cv2.circle(frame, (x, y), 5, dot_color, -1)

        return frame

    @staticmethod
    def get_corrections(landmarks, current_pose):
        """Compares actual joint angles vs ideal ranges and returns correction tips + bad joint indices."""
        corrections = []
        bad_joints = set()

        l_shldr = YogaAnalyzer._xy(landmarks, 11)
        r_shldr = YogaAnalyzer._xy(landmarks, 12)
        l_elbow = YogaAnalyzer._xy(landmarks, 13)
        r_elbow = YogaAnalyzer._xy(landmarks, 14)
        l_wrist = YogaAnalyzer._xy(landmarks, 15)
        r_wrist = YogaAnalyzer._xy(landmarks, 16)
        l_hip   = YogaAnalyzer._xy(landmarks, 23)
        r_hip   = YogaAnalyzer._xy(landmarks, 24)
        l_knee  = YogaAnalyzer._xy(landmarks, 25)
        r_knee  = YogaAnalyzer._xy(landmarks, 26)
        l_ankle = YogaAnalyzer._xy(landmarks, 27)
        r_ankle = YogaAnalyzer._xy(landmarks, 28)
        nose    = YogaAnalyzer._xy(landmarks, 0)

        l_elbow_angle = YogaAnalyzer.calculate_angle(l_shldr, l_elbow, l_wrist)
        r_elbow_angle = YogaAnalyzer.calculate_angle(r_shldr, r_elbow, r_wrist)
        l_knee_angle  = YogaAnalyzer.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle  = YogaAnalyzer.calculate_angle(r_hip, r_knee, r_ankle)
        shoulder_center = YogaAnalyzer._avg(l_shldr, r_shldr)
        hip_center      = YogaAnalyzer._avg(l_hip, r_hip)

        if "Urdhva Hastasana" in current_pose:
            if l_elbow_angle < 160:
                corrections.append("Straighten your LEFT arm")
                bad_joints.update([13, 15])
            if r_elbow_angle < 160:
                corrections.append("Straighten your RIGHT arm")
                bad_joints.update([14, 16])
            if l_wrist[1] > nose[1]:
                corrections.append("Raise LEFT hand above your head")
                bad_joints.add(15)
            if r_wrist[1] > nose[1]:
                corrections.append("Raise RIGHT hand above your head")
                bad_joints.add(16)
            if l_knee_angle < 165:
                corrections.append("Straighten your LEFT leg")
                bad_joints.update([25, 27])
            if r_knee_angle < 165:
                corrections.append("Straighten your RIGHT leg")
                bad_joints.update([26, 28])

        elif "Vrksasana" in current_pose:
            if l_elbow_angle < 160:
                corrections.append("Extend LEFT arm fully overhead")
                bad_joints.update([13, 15])
            if r_elbow_angle < 160:
                corrections.append("Extend RIGHT arm fully overhead")
                bad_joints.update([14, 16])

        elif "Virabhadrasana" in current_pose:
            if l_elbow_angle < 155:
                corrections.append("Extend LEFT arm straight to the side")
                bad_joints.update([13, 15])
            if r_elbow_angle < 155:
                corrections.append("Extend RIGHT arm straight to the side")
                bad_joints.update([14, 16])
            bent_knee = min(l_knee_angle, r_knee_angle)
            if bent_knee > 110:
                corrections.append("Bend your front knee deeper (aim ~90 deg)")
                if l_knee_angle < r_knee_angle:
                    bad_joints.add(25)
                else:
                    bad_joints.add(26)

        elif "T Pose" in current_pose:
            if l_elbow_angle < 155:
                corrections.append("Extend LEFT arm fully horizontal")
                bad_joints.update([13, 15])
            if r_elbow_angle < 155:
                corrections.append("Extend RIGHT arm fully horizontal")
                bad_joints.update([14, 16])

        elif "Slouching" in current_pose:
            corrections.append("Lift your head — tuck your chin back")
            corrections.append("Pull your shoulders back and down")
            bad_joints.update([0, 11, 12])

        # General torso alignment check (applies to all poses)
        if abs(shoulder_center[0] - hip_center[0]) > 0.20:
            corrections.append("Center your torso — reduce side lean")
            bad_joints.update([11, 12, 23, 24])

        if not corrections:
            corrections.append("Great form! Hold the pose.")

        return corrections, bad_joints

    @staticmethod
    def detect_posture(landmarks):
        """Analyzes landmarks and returns a yoga-posture label."""
        nose = YogaAnalyzer._xy(landmarks, 0)
        l_shldr = YogaAnalyzer._xy(landmarks, 11)
        r_shldr = YogaAnalyzer._xy(landmarks, 12)
        l_elbow = YogaAnalyzer._xy(landmarks, 13)
        r_elbow = YogaAnalyzer._xy(landmarks, 14)
        l_wrist = YogaAnalyzer._xy(landmarks, 15)
        r_wrist = YogaAnalyzer._xy(landmarks, 16)
        l_hip = YogaAnalyzer._xy(landmarks, 23)
        r_hip = YogaAnalyzer._xy(landmarks, 24)
        l_knee = YogaAnalyzer._xy(landmarks, 25)
        r_knee = YogaAnalyzer._xy(landmarks, 26)
        l_ankle = YogaAnalyzer._xy(landmarks, 27)
        r_ankle = YogaAnalyzer._xy(landmarks, 28)

        l_elbow_angle = YogaAnalyzer.calculate_angle(l_shldr, l_elbow, l_wrist)
        r_elbow_angle = YogaAnalyzer.calculate_angle(r_shldr, r_elbow, r_wrist)
        l_knee_angle = YogaAnalyzer.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = YogaAnalyzer.calculate_angle(r_hip, r_knee, r_ankle)

        shoulder_center = YogaAnalyzer._avg(l_shldr, r_shldr)
        hip_center = YogaAnalyzer._avg(l_hip, r_hip)
        torso_upright = abs(shoulder_center[0] - hip_center[0]) < 0.20

        wrists_above_head = l_wrist[1] < nose[1] and r_wrist[1] < nose[1]
        wrists_shoulder_level = abs(l_wrist[1] - l_shldr[1]) < 0.15 and abs(r_wrist[1] - r_shldr[1]) < 0.15
        elbows_extended = l_elbow_angle > 135 and r_elbow_angle > 135

        one_knee_bent = (l_knee_angle < 145 and r_knee_angle > 150) or (r_knee_angle < 145 and l_knee_angle > 150)
        both_legs_straight = l_knee_angle > 150 and r_knee_angle > 150
        one_leg_lifted = abs(l_ankle[1] - r_ankle[1]) > 0.08

        if wrists_above_head and elbows_extended and both_legs_straight and torso_upright:
            return "Urdhva Hastasana (Raised Arms)"

        if one_knee_bent and one_leg_lifted and torso_upright:
            return "Vrksasana (Tree Pose)"

        if wrists_shoulder_level and elbows_extended and one_knee_bent:
            return "Virabhadrasana II (Warrior II)"

        if wrists_shoulder_level and elbows_extended and both_legs_straight:
            return "T Pose / Alignment"

        if nose[1] > shoulder_center[1]:
            return "Slouching / Forward Head"

        return "Neutral Standing"

import cv2
from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_name='yolov8n.pt', confidence=0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect_and_draw(self, frame):
        results = self.model.predict(frame, classes=[0], conf=self.confidence, verbose=False)
        
        annotated_frame = frame.copy()
        human_count = 0

        for box in results[0].boxes:
            human_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Person: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame, human_count