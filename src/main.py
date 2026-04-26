import sys
import os
import cv2
import time
import mediapipe as mp
import threading
import queue
import argparse

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None

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


class VoiceAssistant:
    def __init__(self, enabled=True, speak_interval_sec=3.0):
        self.enabled = enabled and pyttsx3 is not None
        self.speak_interval_sec = speak_interval_sec
        self.last_spoken_signature = ""
        self.last_spoken_at = 0.0
        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None

        if not self.enabled:
            if pyttsx3 is None:
                print("Voice assistance is OFF (pyttsx3 not installed).")
            return

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self._worker = threading.Thread(target=self._speaker_loop, daemon=True)
        self._worker.start()
        print("Voice assistance is ON.")

    def _speaker_loop(self):
        while not self._stop_event.is_set():
            try:
                texts = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                for text in texts:
                    if text:
                        self.engine.say(text)
                        self.engine.runAndWait()
            except Exception as err:
                print(f"Voice assistant error: {err}")

    def interrupt(self):
        if not self.enabled:
            return
        try:
            self.engine.stop()
        except Exception:
            pass

    def speak_lines_if_needed(self, lines, min_interval_sec=None, force=False):
        if not self.enabled:
            return

        now = time.time()
        interval = self.speak_interval_sec if min_interval_sec is None else min_interval_sec
        cleaned_lines = [line.replace("—", ", ").strip() for line in lines if line and line.strip()]
        if not cleaned_lines:
            return

        signature = " || ".join(cleaned_lines)

        # Avoid flooding with repeated feedback every frame.
        if (not force) and signature == self.last_spoken_signature and (now - self.last_spoken_at) < interval:
            return

        self.last_spoken_signature = signature
        self.last_spoken_at = now
        # Keep only the latest guidance so stale queued messages are not spoken.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(cleaned_lines)
        except queue.Full:
            pass

    def stop(self):
        if not self.enabled:
            return
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Yoga posture correction with optional voice coaching")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0/1) or video file path")
    parser.add_argument("--voice", action="store_true", help="Enable voice correction coaching (enabled by default)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice correction coaching")
    parser.add_argument("--voice-interval", type=float, default=3.0,
                        help="Minimum seconds between repeated spoken messages")
    return parser.parse_args()


def _resolve_video_source(source_arg):
    if isinstance(source_arg, str) and source_arg.isdigit():
        return int(source_arg)
    return source_arg

def main():
    args = parse_args()

    print("Initializing YOLO Detector...")
    yolo_detector = HumanDetector(model_name='yolov8n.pt', confidence=0.5) 
    
    print("Initializing MediaPipe Logic...")
    ai_engine = AITrackingEngine(max_people=1) 
    analyzer = YogaAnalyzer()
    voice_enabled = not args.no_voice
    voice_assistant = VoiceAssistant(enabled=voice_enabled, speak_interval_sec=args.voice_interval)

    # Open Camera
    print("Connecting to Camera...")
    source = _resolve_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, int) and source != 0:
        print(f"Camera {source} not found. Trying Camera 0...")
        cap = cv2.VideoCapture(0)

    if voice_assistant.enabled:
        voice_assistant.speak_lines_if_needed(["Voice coaching started."], force=True)

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
                # Speak concise correction coaching, skip generic praise to reduce noise.
                spoken_corrections = [tip for tip in corrections if "Great form" not in tip]
                if spoken_corrections:
                    voice_lines = [f"Current pose: {current_pose}"] + spoken_corrections
                    voice_assistant.speak_lines_if_needed(voice_lines, min_interval_sec=args.voice_interval)

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
                voice_assistant.speak_lines_if_needed(["Detecting your pose. Please hold steady."], min_interval_sec=6.0)
        elif human_count > 1:
            cv2.putText(annotated_frame, "Too many people in frame!", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            voice_assistant.speak_lines_if_needed(["Only one person should be in frame."], min_interval_sec=6.0)
        else:
            cv2.putText(annotated_frame, "Waiting for person...", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            voice_assistant.speak_lines_if_needed(["Please step into the camera frame."], min_interval_sec=6.0)

        # Display
        cv2.imshow("Placement Prep Monitor", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    voice_assistant.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()