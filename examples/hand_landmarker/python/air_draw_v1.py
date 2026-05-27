import cv2
import mediapipe as mp
import time
import os
import urllib.request
import numpy as np

# ── Model download ─────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ── mp.tasks setup ─────────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode

# ── Landmark indices ───────────────────────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# ── Colour palette (BGR) ───────────────────────────────────────────────────────
# Each entry: (BGR colour, display name)
PALETTE = [
    ((255, 255, 255), "White"),
    ((0,   255, 180), "Teal"),
    ((80,  255,  80), "Green"),
    ((0,   200, 255), "Yellow"),
    ((80,   80, 255), "Red"),
    ((255,  80,  80), "Blue"),
    ((255,   0, 200), "Purple"),
    ((0,   140, 255), "Orange"),
]

LINE_THICKNESS = 4
CURSOR_RADIUS  = 20

# ── Palette bar geometry ───────────────────────────────────────────────────────
SWATCH_SIZE    = 48   # width & height of each colour swatch
SWATCH_PADDING = 8    # gap between swatches
BAR_TOP        = 10   # y offset from top of frame
# x start is computed at runtime once we know the frame width

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def count_fingers(lm, handedness_label: str) -> int:
    """Count how many fingers are extended."""
    count = 0
    if handedness_label == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


def draw_skeleton(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (80, 80, 80), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (180, 180, 180), -1)


def swatch_rects(fw: int):
    """
    Return a list of (x1, y1, x2, y2) bounding boxes for each swatch,
    centred horizontally across the top of the frame.
    """
    n     = len(PALETTE)
    total = n * SWATCH_SIZE + (n - 1) * SWATCH_PADDING
    x0    = (fw - total) // 2
    rects = []
    for i in range(n):
        x1 = x0 + i * (SWATCH_SIZE + SWATCH_PADDING)
        y1 = BAR_TOP
        rects.append((x1, y1, x1 + SWATCH_SIZE, y1 + SWATCH_SIZE))
    return rects


def draw_palette(frame, rects, selected_idx: int, hover_idx: int):
    """Draw the colour palette bar onto the frame."""
    for i, (x1, y1, x2, y2) in enumerate(rects):
        colour, name = PALETTE[i]

        # Swatch fill
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)

        if i == selected_idx:
            # Thick white border = currently selected
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (255, 255, 255), 3)
            # Label below
            cv2.putText(frame, name, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        elif i == hover_idx:
            # Thin yellow border = fingertip is hovering here
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          (0, 220, 255), 2)

        # Keyboard shortcut hint (1-based)
        cv2.putText(frame, str(i + 1), (x1 + 4, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)


def hit_swatch(tip_x: int, tip_y: int, rects) -> int:
    """Return swatch index if fingertip is inside one, else -1."""
    for i, (x1, y1, x2, y2) in enumerate(rects):
        # Give a small extra hit zone so it's easy to grab
        if x1 - 4 <= tip_x <= x2 + 4 and y1 - 4 <= tip_y <= y2 + 16:
            return i
    return -1


def draw_instructions(frame):
    lines = [
        "1 finger = draw",
        "2+ fingers = pause",
        "Hover fingertip over swatch to pick colour",
        "1-8 keys = pick colour",
        "C = clear  |  Q = quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (12, frame.shape[0] - 16 - (len(lines) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (200, 200, 200), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    canvas = {
        "img":         None,
        "prev":        {},      # last tip position per hand index
        "colour_idx":  0,       # currently selected palette colour (shared)
    }

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.9,
        min_hand_presence_confidence=0.9,
        min_tracking_confidence=1.0,
    )

    # Hover needs to persist briefly so a slow hover still registers
    hover_idx       = -1
    hover_frames    = 0
    HOVER_THRESHOLD = 5   # frames fingertip must stay on a swatch to select it

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            fh, fw, _ = frame.shape

            if canvas["img"] is None:
                canvas["img"] = np.zeros((fh, fw, 3), dtype=np.uint8)

            rects = swatch_rects(fw)

            # ── Detection ─────────────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            active_hand_indices = set()
            current_hover       = -1   # any hand hovering this frame

            if result.hand_landmarks:
                for hand_idx, (lm, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    label = handedness[0].category_name
                    tip_x = int(lm[8].x * fw)
                    tip_y = int(lm[8].y * fh)
                    n_fin = count_fingers(lm, label)

                    draw_skeleton(frame, lm)

                    # ── Colour selection: hover over a swatch ─────────────────
                    hit = hit_swatch(tip_x, tip_y, rects)
                    if hit != -1:
                        current_hover = hit
                        if hit == hover_idx:
                            hover_frames += 1
                            if hover_frames >= HOVER_THRESHOLD:
                                canvas["colour_idx"] = hit
                                hover_frames = 0
                        else:
                            hover_idx    = hit
                            hover_frames = 1
                    
                    # ── Drawing ───────────────────────────────────────────────
                    draw_colour = PALETTE[canvas["colour_idx"]][0]

                    if n_fin == 1 and hit == -1:
                        # Pen down — only when NOT hovering palette
                        active_hand_indices.add(hand_idx)
                        if hand_idx in canvas["prev"]:
                            px, py = canvas["prev"][hand_idx]
                            cv2.line(canvas["img"], (px, py), (tip_x, tip_y),
                                     draw_colour, LINE_THICKNESS)
                        canvas["prev"][hand_idx] = (tip_x, tip_y)

                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS, draw_colour, -1)
                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS + 3, (255, 255, 255), 1)
                    else:
                        canvas["prev"].pop(hand_idx, None)
                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS, (100, 100, 100), 1)

            # Reset hover counter if no hand is over any swatch this frame
            if current_hover == -1:
                hover_idx    = -1
                hover_frames = 3

            # Clear stale hands
            for k in set(canvas["prev"].keys()) - active_hand_indices:
                canvas["prev"].pop(k, None)

            # ── Composite canvas onto webcam frame ────────────────────────────
            gray = cv2.cvtColor(canvas["img"], cv2.COLOR_BGR2GRAY)
            _, mask     = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv    = cv2.bitwise_not(mask)
            frame_bg    = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg   = cv2.bitwise_and(canvas["img"], canvas["img"], mask=mask)
            frame       = cv2.add(frame_bg, canvas_fg)

            # ── UI overlays ───────────────────────────────────────────────────
            draw_palette(frame, rects, canvas["colour_idx"], hover_idx)
            draw_instructions(frame)

            cv2.imshow("Air Draw", frame)

            # ── Keyboard controls ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas["img"]  = np.zeros_like(canvas["img"])
                canvas["prev"] = {}
            elif ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                if idx < len(PALETTE):
                    canvas["colour_idx"] = idx

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
