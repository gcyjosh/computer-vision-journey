"""
Finger Counter using Webcam
----------------------------
Uses MediaPipe Hands to detect and count raised fingers in real time.

Requirements:
    pip install opencv-python mediapipe

Run:
    python finger_counter.py

Controls:
    Q  — quit
    F  — toggle fullscreen
"""

import cv2
import mediapipe as mp

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Landmark indices for each finger tip and its PIP joint (one joint below)
#   Thumb uses a different axis check (x instead of y)
FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_PIPS = [3, 6, 10, 14, 18]   # joints just below each tip


def count_fingers(hand_landmarks, handedness_label: str) -> int:
    """Return the number of extended fingers for one detected hand."""
    lm = hand_landmarks.landmark
    count = 0

    # ── Thumb (horizontal comparison) ────────────────────────────────────────
    # For a right hand: tip is to the LEFT of the IP joint when extended.
    # For a left  hand: tip is to the RIGHT.
    tip_x = lm[FINGER_TIPS[0]].x
    pip_x = lm[FINGER_PIPS[0]].x
    if handedness_label == "Right":
        if tip_x < pip_x:
            count += 1
    else:
        if tip_x > pip_x:
            count += 1

    # ── Four fingers (vertical comparison) ───────────────────────────────────
    # Tip y-coordinate is LESS than PIP y-coordinate when finger is up
    # (MediaPipe y increases downward).
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip_id].y < lm[pip_id].y:
            count += 1

    return count


def draw_overlay(frame, results, finger_counts: list[int]) -> None:
    """Draw hand landmarks and the finger-count overlay onto the frame."""
    h, w, _ = frame.shape
    total = sum(finger_counts)

    # ── Hand landmarks ────────────────────────────────────────────────────────
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style(),
            )

    # ── Total count badge (top-centre) ────────────────────────────────────────
    badge_text = str(total)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 4.0
    thickness  = 8
    (tw, th), baseline = cv2.getTextSize(badge_text, font, scale, thickness)
    bx = (w - tw) // 2
    by = th + 20

    # Shadow
    cv2.putText(frame, badge_text, (bx + 3, by + 3),
                font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, badge_text, (bx, by),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # ── Per-hand label ────────────────────────────────────────────────────────
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_lm, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            label = handedness.classification[0].label  # "Left" / "Right"
            wrist = hand_lm.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            info   = f"{label}: {finger_counts[idx]}"
            cv2.putText(frame, info, (cx - 40, cy - 20),
                        font, 0.9, (0, 255, 180), 2, cv2.LINE_AA)

    # ── Instructions ──────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: quit", (10, h - 15),
                font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that it is connected "
                           "and not in use by another application.")

    fullscreen = False
    window     = "Finger Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame — exiting.")
                break

            # Mirror the frame so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # MediaPipe works on RGB
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Count fingers per detected hand
            finger_counts: list[int] = []
            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_lm, handedness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    label = handedness.classification[0].label
                    finger_counts.append(count_fingers(hand_lm, label))

            draw_overlay(frame, result, finger_counts)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, flag)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
