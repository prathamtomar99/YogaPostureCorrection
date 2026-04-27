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


# class VoiceAssistant:
#     def __init__(self, enabled=True, speak_interval_sec=3.0):
#         self.enabled = enabled and pyttsx3 is not None
#         self.speak_interval_sec = speak_interval_sec
#         self.last_spoken_signature = ""
#         self.last_spoken_at = 0.0
#         self._queue = queue.Queue(maxsize=1)
#         self._stop_event = threading.Event()
#         self._worker = None

#         if not self.enabled:
#             if pyttsx3 is None:
#                 print("Voice assistance is OFF (pyttsx3 not installed).")
#             return

#         self.engine = pyttsx3.init()
#         self.engine.setProperty("rate", 165)
#         self._worker = threading.Thread(target=self._speaker_loop, daemon=True)
#         self._worker.start()
#         print("Voice assistance is ON.")

#     def _speaker_loop(self):
#         while not self._stop_event.is_set():
#             try:
#                 texts = self._queue.get(timeout=0.2)
#             except queue.Empty:
#                 continue

#             try:
#                 for text in texts:
#                     if text:
#                         self.engine.say(text)
#                         self.engine.runAndWait()
#             except Exception as err:
#                 print(f"Voice assistant error: {err}")

#     def interrupt(self):
#         if not self.enabled:
#             return
#         try:
#             self.engine.stop()
#         except Exception:
#             pass

#     def speak_lines_if_needed(self, lines, min_interval_sec=None, force=False):
#         if not self.enabled:
#             return

#         now = time.time()
#         interval = self.speak_interval_sec if min_interval_sec is None else min_interval_sec
#         cleaned_lines = [line.replace("—", ", ").strip() for line in lines if line and line.strip()]
#         if not cleaned_lines:
#             return

#         signature = " || ".join(cleaned_lines)

#         # Avoid flooding with repeated feedback every frame.
#         if (not force) and signature == self.last_spoken_signature and (now - self.last_spoken_at) < interval:
#             return

#         self.last_spoken_signature = signature
#         self.last_spoken_at = now
#         # Keep only the latest guidance so stale queued messages are not spoken.
#         while not self._queue.empty():
#             try:
#                 self._queue.get_nowait()
#             except queue.Empty:
#                 break
#         try:
#             self._queue.put_nowait(cleaned_lines)
#         except queue.Full:
#             pass

#     def stop(self):
#         if not self.enabled:
#             return
#         self._stop_event.set()
#         if self._worker is not None:
#             self._worker.join(timeout=1.0)
import asyncio
import edge_tts
import tempfile
import subprocess
import ctypes

class VoiceAssistant:
    def __init__(self, enabled=True, speak_interval_sec=2.0, voice="en-IN-PrabhatNeural"):
        self.enabled = enabled
        self.speak_interval_sec = speak_interval_sec
        self.voice = voice

        self.last_spoken_at = 0.0
        self.last_requested_text = ""
        self._interrupt_event = threading.Event()
        self._active_process = None
        self._mci_alias = "tts_audio"
        self._is_playing = False

        # Queue for latest message
        self._queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

        print("Edge TTS Voice Assistant is ON." if self.enabled else "Voice OFF.")

    # --------------------------
    # Async speech function
    # --------------------------
    async def _speak_async(self, text):
        filename = None
        try:
            self._is_playing = True
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                filename = f.name

            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            await communicate.save(filename)

            # Play audio (cross-platform)
            if sys.platform == "win32":
                self._play_mp3_windows(filename)
            elif sys.platform == "darwin":
                self._play_with_process(["afplay", filename])
            else:
                self._play_with_process(["mpg123", "-q", filename])

        except Exception as e:
            print(f"[Voice Error]: {e}")
        finally:
            self._is_playing = False
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass

    def _play_with_process(self, command):
        self._active_process = subprocess.Popen(command)
        try:
            while not self._stop_event.is_set() and not self._interrupt_event.is_set():
                if self._active_process.poll() is not None:
                    break
                time.sleep(0.05)
        finally:
            if self._active_process and self._active_process.poll() is None:
                self._active_process.terminate()
                try:
                    self._active_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._active_process.kill()
            self._active_process = None

    def _mci_send(self, command, expect_response=False):
        mci = ctypes.windll.winmm.mciSendStringW
        if expect_response:
            buffer = ctypes.create_unicode_buffer(128)
            err = mci(command, buffer, 128, None)
            return err, buffer.value
        err = mci(command, None, 0, None)
        return err, ""

    def _play_mp3_windows(self, filename):
        alias = self._mci_alias

        # Reset alias if it was left open from a prior failure.
        self._mci_send(f"close {alias}")

        open_cmd = f'open "{filename}" type mpegvideo alias {alias}'
        err, _ = self._mci_send(open_cmd)
        if err != 0:
            raise RuntimeError("Could not open audio file for playback")

        try:
            err, _ = self._mci_send(f"play {alias}")
            if err != 0:
                raise RuntimeError("Could not play audio file")

            while not self._stop_event.is_set() and not self._interrupt_event.is_set():
                err, mode = self._mci_send(f"status {alias} mode", expect_response=True)
                if err != 0 or mode.lower() != "playing":
                    break
                time.sleep(0.05)

            if self._interrupt_event.is_set() or self._stop_event.is_set():
                self._mci_send(f"stop {alias}")
        finally:
            self._mci_send(f"close {alias}")

    # --------------------------
    # Worker thread with event loop
    # --------------------------
    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._interrupt_event.clear()
            loop.run_until_complete(self._speak_async(text))

    # --------------------------
    # Public API
    # --------------------------
    def interrupt(self, clear_queue=False):
        if not self.enabled:
            return

        self._interrupt_event.set()

        if sys.platform == "win32":
            alias = self._mci_alias
            self._mci_send(f"stop {alias}")
            self._mci_send(f"close {alias}")
        elif self._active_process and self._active_process.poll() is None:
            self._active_process.terminate()

        if clear_queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def speak(self, text, force=False):
        if not self.enabled or not text:
            return

        now = time.time()
        is_same_text = text == self.last_requested_text

        # Throttle repetitions, but allow immediate preemption if guidance changed.
        if not force and is_same_text and (now - self.last_spoken_at) < self.speak_interval_sec:
            return

        self.last_spoken_at = now
        self.last_requested_text = text

        # Keep only one upcoming message. Do not preempt active speech unless forced.
        if force:
            self.interrupt(clear_queue=True)
        else:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        if (not force) and self._is_playing:
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass
            return

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def speak_lines_if_needed(self, lines, min_interval_sec=None, force=False):
        if not lines:
            return

        cleaned_lines = [line.replace("—", ", ").strip() for line in lines if line and line.strip()]
        if not cleaned_lines:
            return

        original_interval = self.speak_interval_sec
        if min_interval_sec is not None:
            self.speak_interval_sec = min_interval_sec

        try:
            combined_text = ". ".join(cleaned_lines)
            self.speak(combined_text, force=force)
        finally:
            self.speak_interval_sec = original_interval

    def stop(self):
        self.interrupt(clear_queue=True)
        self._stop_event.set()
        self._worker.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Yoga posture correction with optional voice coaching")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0/1) or video file path")
    parser.add_argument("--voice", action="store_true", help="Enable voice correction coaching (enabled by default)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice correction coaching")
    parser.add_argument("--voice-interval", type=float, default=3.0,
                        help="Minimum seconds between repeated spoken messages")
    parser.add_argument(
        "--voice-perspective",
        choices=["mirror", "anatomical"],
        default="mirror",
        help="Use 'mirror' to swap left/right in spoken coaching for selfie view"
    )
    return parser.parse_args()


def _resolve_video_source(source_arg):
    if isinstance(source_arg, str) and source_arg.isdigit():
        return int(source_arg)
    return source_arg


def _clean_pose_name(current_pose):
    if "(" in current_pose:
        return current_pose.split("(", 1)[0].strip()
    return current_pose.strip()


def _swap_left_right_words(text):
    # Swap side words for mirrored camera coaching while preserving sentence readability.
    tmp = text.replace("LEFT", "__TMP_LEFT__").replace("RIGHT", "LEFT")
    tmp = tmp.replace("__TMP_LEFT__", "RIGHT")
    tmp = tmp.replace("left", "__tmp_left__").replace("right", "left")
    return tmp.replace("__tmp_left__", "right")


def _get_actionable_corrections(corrections, mirror_perspective=True, max_corrections=2):
    cleaned = [c.strip() for c in corrections if c and c.strip()]
    if mirror_perspective:
        cleaned = [_swap_left_right_words(c) for c in cleaned]
    actionable = [c for c in cleaned if "Great form" not in c]
    return actionable[:max_corrections]


class PoseVoiceCoach:
    def __init__(self, correction_delay_sec=1.2, correction_repeat_sec=4.0):
        self.correction_delay_sec = correction_delay_sec
        self.correction_repeat_sec = correction_repeat_sec
        self.active_pose = None
        self.pose_started_at = 0.0
        self.last_correction_signature = ""
        self.last_correction_spoken_at = 0.0
        self.had_actionable_corrections = False
        self.good_form_announced = False

    def reset(self):
        self.active_pose = None
        self.pose_started_at = 0.0
        self.last_correction_signature = ""
        self.last_correction_spoken_at = 0.0
        self.had_actionable_corrections = False
        self.good_form_announced = False

    def next_message(self, current_pose, corrections, mirror_perspective=True):
        now = time.time()
        pose_name = _clean_pose_name(current_pose)

        if pose_name != self.active_pose:
            self.active_pose = pose_name
            self.pose_started_at = now
            self.last_correction_signature = ""
            self.last_correction_spoken_at = 0.0
            self.had_actionable_corrections = False
            self.good_form_announced = False
            return f"Pose switched to {pose_name}."

        if (now - self.pose_started_at) < self.correction_delay_sec:
            return None

        selected = _get_actionable_corrections(corrections, mirror_perspective=mirror_perspective)
        if not selected:
            if self.had_actionable_corrections and not self.good_form_announced:
                self.good_form_announced = True
                return "Good form. Hold the pose."
            return None

        self.had_actionable_corrections = True
        self.good_form_announced = False

        signature = " || ".join(selected)
        should_repeat = (now - self.last_correction_spoken_at) >= self.correction_repeat_sec
        if signature == self.last_correction_signature and not should_repeat:
            return None

        self.last_correction_signature = signature
        self.last_correction_spoken_at = now
        if len(selected) == 1:
            return f"Correction: {selected[0]}."
        return f"Corrections: {selected[0]}. Also, {selected[1]}."

def main():
    args = parse_args()

    print("Initializing YOLO Detector...")
    yolo_detector = HumanDetector(model_name='yolov8n.pt', confidence=0.5) 
    
    print("Initializing MediaPipe Logic...")
    ai_engine = AITrackingEngine(max_people=1) 
    analyzer = YogaAnalyzer()
    voice_enabled = not args.no_voice
    voice_assistant = VoiceAssistant(enabled=voice_enabled, speak_interval_sec=args.voice_interval)
    pose_voice_coach = PoseVoiceCoach(correction_delay_sec=1.2, correction_repeat_sec=max(4.0, args.voice_interval))

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
                voice_message = pose_voice_coach.next_message(
                    current_pose,
                    corrections,
                    mirror_perspective=(args.voice_perspective == "mirror")
                )
                if voice_message:
                    voice_assistant.speak_lines_if_needed([voice_message], min_interval_sec=args.voice_interval)

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
                pose_voice_coach.reset()
                voice_assistant.speak_lines_if_needed(["Detecting your pose. Please hold steady."], min_interval_sec=6.0)
        elif human_count > 1:
            cv2.putText(annotated_frame, "Too many people in frame!", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            pose_voice_coach.reset()
            voice_assistant.speak_lines_if_needed(["Only one person should be in frame."], min_interval_sec=6.0)
        else:
            cv2.putText(annotated_frame, "Waiting for person...", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            pose_voice_coach.reset()
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