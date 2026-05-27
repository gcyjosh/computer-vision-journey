"""
Air Draw — fast + stable
=========================
Key fixes vs previous version:
  1. Composite: 4 OpenCV ops replaced with one numpy boolean mask copy
  2. Hue strip resized once at startup, not every frame
  3. Stabiliser window reduced to 8 (smooth but no noticeable lag)
  4. Bézier skipped for very short segments (< 3 px) — no wasted work
"""
import cv2, mediapipe as mp, time, os, urllib.request, math
import numpy as np
from collections import deque

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading model…"); urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode

FINGER_TIPS    = [4, 8, 12, 16, 20]
FINGER_PIPS    = [3, 6, 10, 14, 18]
POINTER_LM     = [6, 7, 8]     # average top 3 index landmarks — stabler than tip alone
LINE_THICKNESS = 5

# ── Tuning ─────────────────────────────────────────────────────────────────────
STABILIZER_WINDOW = 8    # frames to average — raise for smoother, lower for faster feel
MIN_CUTOFF        = 0.5
BETA              = 0.8

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

SLIDER_H      = 30
SLIDER_TOP    = 12
SLIDER_MARGIN = 60
KNOB_R        = 13

# ── One Euro Filter ────────────────────────────────────────────────────────────
class _OEF:
    def __init__(self): self._x=self._dx=self._t=None
    def __call__(self,x,t):
        if self._x is None: self._x=x;self._t=t;self._dx=0.;return x
        dt=max(t-self._t,1e-6);self._t=t
        ad=self._a(1.,dt);self._dx=ad*(x-self._x)/dt+(1-ad)*self._dx
        a=self._a(MIN_CUTOFF+BETA*abs(self._dx),dt)
        self._x=a*x+(1-a)*self._x;return self._x
    @staticmethod
    def _a(c,dt): return 1/(1+1/(2*math.pi*c*dt))
    def reset(self): self._x=self._dx=self._t=None

class HandFilter:
    def __init__(self): self.fx=_OEF();self.fy=_OEF()
    def smooth(self,x,y,t): return self.fx(x,t),self.fy(y,t)
    def reset(self): self.fx.reset();self.fy.reset()

# ── Stabilised stroke ──────────────────────────────────────────────────────────
class StabilisedStroke:
    def __init__(self):
        self._buf      = deque(maxlen=STABILIZER_WINDOW)
        self._pts      = []
        self._last_pen = None

    def push(self, fx, fy, canvas, colour):
        self._buf.append((fx, fy))
        n   = len(self._buf)
        pen_x = int(sum(p[0] for p in self._buf) / n)
        pen_y = int(sum(p[1] for p in self._buf) / n)
        if self._last_pen == (pen_x, pen_y):
            return pen_x, pen_y
        self._last_pen = (pen_x, pen_y)
        self._pts.append((pen_x, pen_y))
        np_ = len(self._pts)
        if np_ == 2:
            cv2.line(canvas, self._pts[0], self._pts[1],
                     colour, LINE_THICKNESS, cv2.LINE_AA)
        elif np_ >= 3:
            a,b,c = self._pts[-3],self._pts[-2],self._pts[-1]
            # Skip Bézier for tiny segments — saves CPU with no visible difference
            if math.hypot(b[0]-a[0],b[1]-a[1]) < 3:
                cv2.line(canvas, a, c, colour, LINE_THICKNESS, cv2.LINE_AA)
            else:
                m1=((a[0]+b[0])//2,(a[1]+b[1])//2)
                m2=((b[0]+c[0])//2,(b[1]+c[1])//2)
                prev=m1
                for i in range(1,13):
                    t=i/12;mt=1-t
                    p=(int(mt*mt*m1[0]+2*mt*t*b[0]+t*t*m2[0]),
                       int(mt*mt*m1[1]+2*mt*t*b[1]+t*t*m2[1]))
                    cv2.line(canvas,prev,p,colour,LINE_THICKNESS,cv2.LINE_AA)
                    prev=p
        return pen_x, pen_y

    def flush(self, canvas, colour):
        if len(self._pts) >= 2:
            cv2.line(canvas,self._pts[-2],self._pts[-1],
                     colour,LINE_THICKNESS,cv2.LINE_AA)
        self._buf.clear();self._pts=[];self._last_pen=None

# ── HSV helpers ────────────────────────────────────────────────────────────────
def make_hue_strip(w, h):
    strip = np.zeros((h,w,3),np.uint8)
    for x in range(w):
        strip[:,x] = cv2.cvtColor(
            np.uint8([[[int(x/w*179),220,220]]]),cv2.COLOR_HSV2BGR)[0][0]
    return strip

def hue_to_bgr(h):
    b=cv2.cvtColor(np.uint8([[[int(h*179),220,220]]]),cv2.COLOR_HSV2BGR)[0][0]
    return (int(b[0]),int(b[1]),int(b[2]))

def draw_slider(frame, hue, strip_resized):
    # strip_resized is already the right width — no resize needed here
    fw = frame.shape[1]
    x0,x1 = SLIDER_MARGIN, fw-SLIDER_MARGIN
    y0,y1 = SLIDER_TOP, SLIDER_TOP+SLIDER_H
    frame[y0:y1,x0:x1] = strip_resized
    cv2.rectangle(frame,(x0,y0),(x1,y1),(180,180,180),1)
    kx=x0+int(hue*(x1-x0)); ky=(y0+y1)//2
    c=hue_to_bgr(hue)
    cv2.circle(frame,(kx,ky),KNOB_R,(255,255,255),-1)
    cv2.circle(frame,(kx,ky),KNOB_R-3,c,-1)
    cv2.circle(frame,(kx,ky),KNOB_R,(60,60,60),1)
    bx=x0-KNOB_R*2-6
    cv2.rectangle(frame,(bx,y0),(bx+KNOB_R*2,y1),c,-1)
    cv2.rectangle(frame,(bx,y0),(bx+KNOB_R*2,y1),(180,180,180),1)

def slider_hit(tx,ty,fw):
    x0,x1=SLIDER_MARGIN,fw-SLIDER_MARGIN
    if x0<=tx<=x1 and SLIDER_TOP-KNOB_R<=ty<=SLIDER_TOP+SLIDER_H+KNOB_R:
        return max(0.,min(1.,(tx-x0)/(x1-x0)))
    return None

def count_fingers(lm,label):
    c =int(lm[4].x<lm[3].x) if label=="Right" else int(lm[4].x>lm[3].x)
    for tip,pip in zip(FINGER_TIPS[1:],FINGER_PIPS[1:]):
        c+=int(lm[tip].y<lm[pip].y)
    return c

def pointer_pos(lm,fw,fh):
    return (sum(lm[i].x for i in POINTER_LM)/3*fw,
            sum(lm[i].y for i in POINTER_LM)/3*fh)

def draw_skeleton(frame,lm):
    h,w=frame.shape[:2]
    pts=[(int(l.x*w),int(l.y*h)) for l in lm]
    for s,e in HAND_CONNECTIONS: cv2.line(frame,pts[s],pts[e],(70,70,70),1)
    for pt in pts: cv2.circle(frame,pt,3,(150,150,150),-1)

def draw_ui(frame):
    fh=frame.shape[0]
    for i,l in enumerate(["1 finger = draw  |  2+ = pause",
                           "Point at colour bar to change colour",
                           "C = clear  |  Q = quit"]):
        cv2.putText(frame,l,(12,fh-12-(2-i)*22),
                    cv2.FONT_HERSHEY_SIMPLEX,.50,(190,190,190),1,cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cap=cv2.VideoCapture(0)
    cap.set(3,1280);cap.set(4,720)

    canvas       = None
    strip_scaled = None   # hue strip resized once to fit frame width
    filters      = {}
    strokes      = {}
    hue          = 0.55

    opts=HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with HandLandmarker.create_from_options(opts) as det:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            fh,fw=frame.shape[:2]

            if canvas is None:
                canvas       = np.zeros((fh,fw,3),np.uint8)
                # Build and resize the hue strip ONCE
                strip_scaled = cv2.resize(make_hue_strip(360,SLIDER_H),
                                          (fw-SLIDER_MARGIN*2, SLIDER_H))

            t      = time.time()
            rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(t*1000))

            colour   = hue_to_bgr(hue)
            detected = set()

            if result.hand_landmarks:
                for hi,(lm,hand) in enumerate(
                        zip(result.hand_landmarks,result.handedness)):
                    detected.add(hi)
                    label=hand[0].category_name
                    n_fin=count_fingers(lm,label)
                    draw_skeleton(frame,lm)

                    if hi not in filters:
                        filters[hi]=HandFilter();strokes[hi]=None

                    raw_x,raw_y=pointer_pos(lm,fw,fh)
                    fx,fy=filters[hi].smooth(raw_x,raw_y,t)
                    tip_x,tip_y=int(fx),int(fy)

                    h_hit=slider_hit(tip_x,tip_y,fw)
                    if h_hit is not None:
                        hue=h_hit;colour=hue_to_bgr(hue)
                        if strokes.get(hi):
                            strokes[hi].flush(canvas,colour);strokes[hi]=None
                        cv2.circle(frame,(tip_x,tip_y),8,(255,255,255),-1)
                        cv2.circle(frame,(tip_x,tip_y),8,(60,60,60),1)
                        continue

                    if n_fin==1:
                        if strokes[hi] is None:
                            strokes[hi]=StabilisedStroke()
                        pen_x,pen_y=strokes[hi].push(fx,fy,canvas,colour)
                        cv2.circle(frame,(tip_x,tip_y),5,(80,80,80),1)
                        cv2.circle(frame,(pen_x,pen_y),10,colour,-1)
                        cv2.circle(frame,(pen_x,pen_y),13,(255,255,255),1)
                    else:
                        if strokes.get(hi):
                            strokes[hi].flush(canvas,colour);strokes[hi]=None
                        cv2.circle(frame,(tip_x,tip_y),10,(90,90,90),1)

            for k in set(filters)-detected:
                if strokes.get(k): strokes[k].flush(canvas,hue_to_bgr(hue))
                filters.pop(k,None);strokes.pop(k,None)

            # ── Fast composite: single numpy copy instead of 4 OpenCV ops ─────
            # canvas pixels that are non-zero overwrite the webcam frame.
            # np.any(..., axis=2) gives a (H,W) boolean mask in one shot.
            mask = np.any(canvas > 0, axis=2)
            frame[mask] = canvas[mask]

            draw_slider(frame,hue,strip_scaled)
            draw_ui(frame)
            cv2.imshow("Air Draw",frame)

            k=cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif k==ord('c'):
                canvas[:]=0
                for s in strokes.values():
                    if s: s.flush(canvas,hue_to_bgr(hue))
                strokes={k:None for k in strokes}
                for f in filters.values(): f.reset()

    cap.release();cv2.destroyAllWindows()

if __name__=="__main__": main()
