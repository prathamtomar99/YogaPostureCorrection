import sys
import os
import cv2
import time
import mediapipe as mp

# Add current directory to the Python path to bypass the ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# import the logic blocks
try:
    from engine import HumanDetector
    from engine import AITrackingEngine, YogaAnalyzer
except ModuleNotFoundError as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Please make sure 'humanDetector.py' and 'engine.py' are in the 'src/' folder.")
    sys.exit(1)

def main():
    print("Initializing YOLO Detector...")
    yolo_detector = HumanDetector(model_name='yolov8n.pt', confidence=0.5) 
    
    print("Initializing MediaPipe Logic...")
    ai_engine = AITrackingEngine(max_people=1) 
    analyzer = YogaAnalyzer()

    # Open Camera
    print("Connecting to Camera...")
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened():
        print("Camera 1 not found. Trying Camera 0...")
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        frame = cv2.flip(frame, 1)

        # 1. Human Detect (YOLO)
        annotated_frame, human_count = yolo_detector.detect_and_draw(frame)
        
        cv2.putText(annotated_frame, f"Total Humans: {human_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Keypoints + Posture (MediaPipe)
        if human_count == 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            mp_person_count, landmarks = ai_engine.get_person_count_and_landmarks(mp_image, timestamp_ms)

            if landmarks and mp_person_count >= 1:
                current_pose = analyzer.detect_posture(landmarks)
                corrections, bad_joints = analyzer.get_corrections(landmarks, current_pose)

                pose_color = (0, 0, 255) if "Slouching" in current_pose else (0, 255, 0)
                cv2.putText(annotated_frame, f"Pose: {current_pose}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, pose_color, 2)

                for i, tip in enumerate(corrections):
                    tip_color = (0, 200, 0) if "Great" in tip else (0, 100, 255)
                    cv2.putText(annotated_frame, tip, (20, 120 + i * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tip_color, 2)

                annotated_frame = analyzer.draw_landmarks_and_skeleton(annotated_frame, landmarks, bad_joints=bad_joints)
            else:
                cv2.putText(annotated_frame, "Detecting pose...", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif human_count > 1:
            cv2.putText(annotated_frame, "Too many people in frame!", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, "Waiting for person...", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display
        cv2.imshow("Placement Prep Monitor", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()