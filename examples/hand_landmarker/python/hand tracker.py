import cv2
import mediapipe as mp
import time 
import urllib.request
import os

BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode          = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~8 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

LANDMARK_COLOUR   = (0, 255, 180)   # teal dots
CONNECTION_COLOUR = (255, 255, 255)  # white lines
LANDMARK_RADIUS   = 5
CONNECTION_WIDTH  = 5

def draw_landmarks(frame, hand_landmarks_list):
    """
    Manually draw all detected hands onto the frame.

    hand_landmarks_list is a list of hands; each hand is a list of 21
    NormalizedLandmark objects with .x / .y in [0, 1] relative to frame size.
    We multiply by (w, h) to get pixel coordinates.
    """
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert normalised coords → pixel coords once for all 21 points
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        # Draw skeleton lines first (so dots appear on top)
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx],
                     CONNECTION_COLOUR, CONNECTION_WIDTH)

        # Draw landmark dots
        for pt in points:
            cv2.circle(frame, pt, LANDMARK_RADIUS, LANDMARK_COLOUR, -1)
            
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
def main():
    cap = cv2.VideoCapture(1)
    cap.set(3,1280)
    cap.set(4,720)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while True: # each iteration grabs a fresh wencam frame
            attempt = 0 
            success, img = cap.read()
            while not success and attempt < 5: # retry loop, used when webcam fails temporarily
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
            if not success: 
                print("Failed to read frame")
                break

            img = cv2.flip(img,1)
            h,w,_ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            if result.hand_landmarks:
                draw_landmarks(img, result.hand_landmarks)

            cv2.imshow("Image",img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()