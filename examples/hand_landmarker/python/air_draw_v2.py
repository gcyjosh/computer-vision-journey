import cv2
import mediapipe as mp
import time, os, urllib.request
import numpy as np
from collections import deque

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

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
CURSOR_RADIUS  = 10

# ── Smoothing knobs ────────────────────────────────────────────────────────────
EMA_ALPHA  = 0.25   # lower = smoother EMA (first-pass noise reduction)
DEAD_ZONE  = 4      # pixels — ignore movement smaller than this (kills micro-jitter)
BUF_SIZE   = 8      # how many recent points to keep for Bézier smoothing

SWATCH_SIZE    = 48
SWATCH_PADDING = 8
BAR_TOP        = 10

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


# ── Smoothing helpers ──────────────────────────────────────────────────────────

def ema_smooth(raw_x, raw_y, hand_idx, ema_state):
    """Exponential moving average — reduces high-frequency noise."""
    if hand_idx not in ema_state:
        ema_state[hand_idx] = (float(raw_x), float(raw_y))
    else:
        px, py = ema_state[hand_idx]
        ema_state[hand_idx] = (
            EMA_ALPHA * raw_x + (1 - EMA_ALPHA) * px,
            EMA_ALPHA * raw_y + (1 - EMA_ALPHA) * py,
        )
    sx, sy = ema_state[hand_idx]
    return sx, sy   # return floats, keep precision


def bezier_points(p0, p1, p2, steps=16):
    """
    Quadratic Bézier from p0 → p2 with p1 as the control point.
    Returns `steps` interpolated (x, y) integer points along the curve.

    This is the key to smooth drawing:
      Instead of line(prev, curr) every frame, we pick three consecutive
      buffered points and draw a curve *through* them.  Small wobbles in
      individual points get averaged out by the curve shape.
    """
    pts = []
    for i in range(steps + 1):
        t  = i / steps
        mt = 1 - t
        x  = mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0]
        y  = mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1]
        pts.append((int(x), int(y)))
    return pts


def flush_buffer_to_canvas(buf, canvas_img, colour):
    """
    Draw smooth Bézier curves through the buffered stroke points.

    Technique (same as HTML5 Canvas smooth drawing):
      For every consecutive triple of points A, B, C:
        - Compute midpoints M1 = mid(A,B)  and  M2 = mid(B,C)
        - Draw a quadratic Bézier from M1 to M2 with B as control point
      This guarantees C1 continuity — the curves join smoothly with no corners.
    """
    if len(buf) < 3:
        # Not enough points for a curve yet — fall back to a straight line
        if len(buf) == 2:
            cv2.line(canvas_img, buf[0], buf[1], colour, LINE_THICKNESS)
        return

    for i in range(len(buf) - 2):
        a = buf[i]
        b = buf[i + 1]
        c = buf[i + 2]
        m1 = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        m2 = ((b[0] + c[0]) // 2, (b[1] + c[1]) // 2)
        curve = bezier_points(m1, b, m2)
        for j in range(len(curve) - 1):
            cv2.line(canvas_img, curve[j], curve[j+1], colour, LINE_THICKNESS)


# ── MediaPipe / drawing helpers ────────────────────────────────────────────────

def count_fingers(lm, label):
    count = 0
    if label == "Right":
        if lm[4].x < lm[3].x: count += 1
    else:
        if lm[4].x > lm[3].x: count += 1
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip].y < lm[pip].y: count += 1
    return count


def draw_skeleton(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (80,80,80), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (180,180,180), -1)


def swatch_rects(fw):
    n = len(PALETTE)
    total = n * SWATCH_SIZE + (n-1) * SWATCH_PADDING
    x0 = (fw - total) // 2
    return [(x0 + i*(SWATCH_SIZE+SWATCH_PADDING), BAR_TOP,
             x0 + i*(SWATCH_SIZE+SWATCH_PADDING) + SWATCH_SIZE, BAR_TOP+SWATCH_SIZE)
            for i in range(n)]


def draw_palette(frame, rects, sel, hover):
    for i, (x1,y1,x2,y2) in enumerate(rects):
        c, name = PALETTE[i]
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, -1)
        if i == sel:
            cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), (255,255,255), 3)
            cv2.putText(frame, name, (x1, y2+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        elif i == hover:
            cv2.rectangle(frame, (x1-2,y1-2), (x2+2,y2+2), (0,220,255), 2)
        cv2.putText(frame, str(i+1), (x1+4,y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)


def hit_swatch(x, y, rects):
    for i,(x1,y1,x2,y2) in enumerate(rects):
        if x1-4 <= x <= x2+4 and y1-4 <= y <= y2+16:
            return i
    return -1


def draw_instructions(frame):
    for i, line in enumerate([
        "1 finger = draw  |  2+ fingers = pause",
        "Hover fingertip over colour to pick  |  keys 1-8",
        "C = clear  |  Q = quit",
    ]):
        cv2.putText(frame, line, (12, frame.shape[0]-16-(2-i)*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    canvas_img  = None
    ema_state   = {}          # hand_idx → (float x, float y)  — EMA state
    stroke_bufs = {}          # hand_idx → deque of (int x, int y) — recent stroke points
    last_drawn  = {}          # hand_idx → (int x, int y) — last point committed to canvas
    colour_idx  = 0
    hover_idx   = -1
    hover_count = 0
    HOVER_THRESHOLD = 8

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fh, fw, _ = frame.shape

            if canvas_img is None:
                canvas_img = np.zeros((fh, fw, 3), dtype=np.uint8)

            rects = swatch_rects(fw)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            active = set()
            current_hover = -1

            if result.hand_landmarks:
                for hi, (lm, hand) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    label = hand[0].category_name
                    n_fin = count_fingers(lm, label)
                    draw_skeleton(frame, lm)

                    # 1. EMA smooth the raw position
                    raw_x = lm[8].x * fw
                    raw_y = lm[8].y * fh
                    sx, sy = ema_smooth(raw_x, raw_y, hi, ema_state)
                    tip_x, tip_y = int(sx), int(sy)

                    # 2. Palette hover check
                    hit = hit_swatch(tip_x, tip_y, rects)
                    if hit != -1:
                        current_hover = hit
                        if hit == hover_idx:
                            hover_count += 1
                            if hover_count >= HOVER_THRESHOLD:
                                colour_idx  = hit
                                hover_count = 0
                        else:
                            hover_idx   = hit
                            hover_count = 1

                    draw_colour = PALETTE[colour_idx][0]

                    if n_fin == 1 and hit == -1:
                        active.add(hi)

                        # 3. Dead zone — only add point if moved enough
                        if hi not in last_drawn or \
                           np.hypot(tip_x - last_drawn[hi][0],
                                    tip_y - last_drawn[hi][1]) >= DEAD_ZONE:

                            if hi not in stroke_bufs:
                                stroke_bufs[hi] = deque(maxlen=BUF_SIZE)
                            stroke_bufs[hi].append((tip_x, tip_y))
                            last_drawn[hi] = (tip_x, tip_y)

                            # 4. Draw Bézier curve through buffered points
                            flush_buffer_to_canvas(
                                list(stroke_bufs[hi]), canvas_img, draw_colour)

                        # Cursor
                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS, draw_colour, -1)
                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS+3, (255,255,255), 1)
                    else:
                        # Pen up — clear this hand's stroke state
                        stroke_bufs.pop(hi, None)
                        last_drawn.pop(hi, None)
                        cv2.circle(frame, (tip_x, tip_y), CURSOR_RADIUS, (100,100,100), 1)

            # Clean up hands that left the frame
            detected = set(range(len(result.hand_landmarks))) if result.hand_landmarks else set()
            for k in set(ema_state) - detected:
                ema_state.pop(k, None)
                stroke_bufs.pop(k, None)
                last_drawn.pop(k, None)

            if current_hover == -1:
                hover_idx   = -1
                hover_count = 0

            # Composite
            gray    = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            inv     = cv2.bitwise_not(mask)
            frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=inv),
                              cv2.bitwise_and(canvas_img, canvas_img, mask=mask))

            draw_palette(frame, rects, colour_idx, hover_idx)
            draw_instructions(frame)
            cv2.imshow("Air Draw", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas_img = np.zeros((fh, fw, 3), dtype=np.uint8)
                stroke_bufs.clear()
                last_drawn.clear()
                ema_state.clear()
            elif ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                if idx < len(PALETTE):
                    colour_idx = idx

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
