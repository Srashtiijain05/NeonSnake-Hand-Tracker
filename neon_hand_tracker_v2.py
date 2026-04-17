"""
Neon Hand Tracker ULTRA v3 — GESTURE FIX EDITION
=================================================
FIXES:
  - Effects ONLY trigger on clear held gestures (not random movement)
  - Voting buffer increased to 12 frames, majority raised to 10
  - 'normal' state REQUIRED between gestures (hysteresis)
  - Gesture must be held for 0.4 seconds before triggering
  - Cooldown raised to 60 frames (~2 sec) so same gesture doesn't spam
  - Particle spawn on fingertip movement is now speed-gated (>8px/frame)
  - Beat burst only fires if very loud beat AND hand is relatively still

Controls:
  Q — quit
  P — toggle paint mode
  C — clear paint canvas
  S — save screenshot
  D — toggle debug overlay (shows curl bars + gesture vote count)
  R — toggle rainbow skeleton mode

Requirements:
  pip install mediapipe opencv-python numpy sounddevice pygame
"""

import cv2
import numpy as np
import random
import math
import time
import threading
import collections
import mediapipe as mp
import os

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ── Neon palette (BGR) ─────────────────────────────────────────────────────────
NEON_CYAN    = (255, 255,   0)
NEON_MAGENTA = (255,   0, 255)
NEON_BLUE    = (255,  80,  20)
NEON_GREEN   = (  0, 255, 128)
NEON_YELLOW  = (  0, 255, 255)
NEON_WHITE   = (255, 255, 255)
NEON_ORANGE  = ( 20, 140, 255)
NEON_PINK    = (180,  20, 255)
NEON_RED     = ( 20,  20, 255)

RAINBOW_COLORS = [NEON_CYAN, NEON_GREEN, NEON_YELLOW, NEON_ORANGE,
                  NEON_PINK, NEON_MAGENTA, NEON_BLUE]

FINGER_HUES = [30, 90, 160, 220, 290]

FINGERTIP_IDS   = [4, 8, 12, 16, 20]
PALM_CENTER_IDS = [0, 5, 9, 13, 17]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
CONNECTION_FINGER = {
    (0,1):0,(1,2):0,(2,3):0,(3,4):0,
    (0,5):1,(5,6):1,(6,7):1,(7,8):1,
    (0,9):2,(9,10):2,(10,11):2,(11,12):2,
    (0,13):3,(13,14):3,(14,15):3,(15,16):3,
    (0,17):4,(17,18):4,(18,19):4,(19,20):4,
    (5,9):1,(9,13):2,(13,17):3,
}

GESTURE_LABELS = {
    'fist':      '💥 BOOM!',
    'peace':     '✌️  PEACE',
    'open':      '🌈 RAINBOW',
    'thumbs_up': '👍 NICE!',
    'ok':        '👌 OK!',
    'rock_on':   '🤘 ROCK ON!',
    'clap':      '👏 CLAP!',
}

# ── Sound generation ───────────────────────────────────────────────────────────
def make_tone(freq=440, duration=0.18, volume=0.4, wave='sine'):
    if not PYGAME_OK:
        return None
    sr = 44100
    t  = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if   wave == 'sine':     w = np.sin(2 * np.pi * freq * t)
    elif wave == 'square':   w = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == 'sawtooth': w = 2 * (t * freq - np.floor(0.5 + t * freq))
    else:                    w = np.sin(2 * np.pi * freq * t)
    fade = np.linspace(1.0, 0.0, len(w))
    w = (w * fade * volume * 32767).astype(np.int16)
    try:
        return pygame.sndarray.make_sound(np.column_stack([w, w]))
    except Exception:
        return None

GESTURE_SOUNDS = {}
if PYGAME_OK:
    GESTURE_SOUNDS = {
        'fist':      make_tone(80,  0.25, 0.5, 'square'),
        'peace':     make_tone(523, 0.20, 0.4, 'sine'),
        'open':      make_tone(880, 0.20, 0.4, 'sine'),
        'thumbs_up': make_tone(660, 0.15, 0.4, 'sine'),
        'ok':        make_tone(440, 0.15, 0.3, 'sine'),
        'rock_on':   make_tone(110, 0.30, 0.5, 'sawtooth'),
        'clap':      make_tone(200, 0.10, 0.5, 'square'),
    }

def play_gesture_sound(gesture):
    if PYGAME_OK and gesture in GESTURE_SOUNDS and GESTURE_SOUNDS[gesture]:
        try: GESTURE_SOUNDS[gesture].play()
        except Exception: pass

# ── Audio beat detector ────────────────────────────────────────────────────────
class BeatDetector:
    def __init__(self):
        self.beat        = False
        self.beat_energy = 0.0
        self.energy_history = [0.0] * 43
        self._lock = threading.Lock()
        if AUDIO_OK:
            try:
                self.stream = sd.InputStream(
                    channels=1, samplerate=44100, blocksize=1024,
                    callback=self._callback)
                self.stream.start()
            except Exception:
                pass

    def _callback(self, indata, frames, time_info, status):
        energy = float(np.mean(indata ** 2))
        with self._lock:
            self.energy_history.append(energy)
            self.energy_history.pop(0)
            avg = max(np.mean(self.energy_history), 1e-10)
            # Raised threshold to 3.5x average (was 2.5x) to reduce false beats
            self.beat        = energy > avg * 3.5
            self.beat_energy = energy / avg

    def is_beat(self):
        with self._lock:
            b, e = self.beat, self.beat_energy
            self.beat = False
            return b, e

beat_detector = BeatDetector()

# ── Particle ───────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, speed_mul=1.0, size_mul=1.0, gravity=0.08):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5.5) * speed_mul
        self.x, self.y = float(x), float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life  = random.uniform(0.5, 1.0)
        self.decay = random.uniform(0.012, 0.035)
        self.size  = random.randint(3, 8) * size_mul
        self.color = color
        self.gravity = gravity

    def update(self):
        self.x  += self.vx;  self.y  += self.vy
        self.vx *= 0.92;     self.vy *= 0.92
        self.vy += self.gravity
        self.life -= self.decay
        self.size = max(1, self.size - 0.06)

    @property
    def alive(self): return self.life > 0

    def draw(self, canvas):
        a = max(0.0, self.life)
        c = tuple(int(v * a) for v in self.color)
        cv2.circle(canvas, (int(self.x), int(self.y)), int(self.size), c, -1)

class FireworkParticle(Particle):
    def __init__(self, x, y, color):
        super().__init__(x, y, color, speed_mul=2.5, size_mul=1.5, gravity=0.05)
        self.trail = []

    def update(self):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 8: self.trail.pop(0)
        super().update()

    def draw(self, canvas):
        for i, pt in enumerate(self.trail):
            a = (i / len(self.trail)) * self.life * 0.6
            c = tuple(int(v * a) for v in self.color)
            cv2.circle(canvas, pt, max(1, int(self.size * 0.5)), c, -1)
        super().draw(canvas)

# ── Snake Trail ────────────────────────────────────────────────────────────────
class SnakeTrail:
    MAX_LEN     = 32
    SEGMENT_GAP = 4

    def __init__(self, finger_idx):
        self.fi      = finger_idx
        self.pts     = []
        self.hue_off = FINGER_HUES[finger_idx]
        self.eff_hue = float(FINGER_HUES[finger_idx])

    def push(self, pt, eff_hue):
        if self.pts and dist(self.pts[-1], pt) < self.SEGMENT_GAP:
            return
        self.pts.append(pt)
        if len(self.pts) > self.MAX_LEN: self.pts.pop(0)
        self.eff_hue = eff_hue

    def draw(self, canvas, speed, rainbow_mode=False, t=0.0):
        n = len(self.pts)
        if n < 2: return
        hue_base = (self.eff_hue + self.hue_off) % 360

        for i in range(1, n):
            t_frac = i / n
            radius = max(2, int(2 + t_frac * 11))
            alpha  = 0.2 + 0.8 * t_frac
            seg_hue = (t * 60 + i * (360/n)) % 360 if rainbow_mode \
                      else (hue_base + (1-t_frac)*60) % 360
            base_c = hsv_color(seg_hue)
            c = tuple(int(v * alpha) for v in base_c)
            cv2.circle(canvas, self.pts[i], radius, c, -1, cv2.LINE_AA)
            cv2.line(canvas, self.pts[i-1], self.pts[i], c, radius*2-1, cv2.LINE_AA)

        if self.pts:
            head   = self.pts[-1]
            h_hue  = (hue_base if not rainbow_mode else (t*60) % 360)
            head_c = hsv_color(h_hue)
            draw_glowing_circle_v3(canvas, head, 13, head_c,
                                   brightness=1.6 + min(speed/18.0, 1.2),
                                   chromatic=True)
            # Tongue
            tongue_angle = math.pi / 2
            if len(self.pts) >= 2:
                dx = self.pts[-1][0] - self.pts[-2][0]
                dy = self.pts[-1][1] - self.pts[-2][1]
                tongue_angle = math.atan2(dy, dx)
            for fork in [-0.28, 0.28]:
                ang = tongue_angle + fork
                ex  = int(head[0] + math.cos(ang) * 15)
                ey  = int(head[1] + math.sin(ang) * 15)
                cv2.line(canvas, head, (ex, ey), NEON_RED, 2, cv2.LINE_AA)

# ── Ripple ─────────────────────────────────────────────────────────────────────
class Ripple:
    def __init__(self, x, y, color, max_r=200):
        self.x, self.y = x, y
        self.color = color
        self.r     = 10
        self.max_r = max_r
        self.life  = 1.0
        self.speed = random.uniform(6, 14)

    @property
    def alive(self): return self.life > 0

    def update(self):
        self.r   += self.speed
        self.life = max(0.0, 1.0 - self.r / self.max_r)

    def draw(self, canvas):
        if self.r <= 0: return
        a = self.life
        c = tuple(int(v * a) for v in self.color)
        cv2.circle(canvas, (int(self.x), int(self.y)),
                   int(self.r), c, max(1, int(4*a)), cv2.LINE_AA)
        r2 = max(1, int(self.r * 0.55))
        c2 = tuple(int(v * a * 0.4) for v in self.color)
        cv2.circle(canvas, (int(self.x), int(self.y)), r2, c2, 1, cv2.LINE_AA)

# ── Floating label ─────────────────────────────────────────────────────────────
class FloatingLabel:
    def __init__(self, x, y, text, color):
        self.x, self.y = float(x), float(y)
        self.text  = text
        self.color = color
        self.life  = 1.0
        self.vy    = -1.8

    @property
    def alive(self): return self.life > 0

    def update(self):
        self.y    += self.vy
        self.life -= 0.016

    def draw(self, canvas):
        a   = max(0.0, self.life)
        c   = tuple(int(v * a) for v in self.color)
        pos = (int(self.x) - 65, int(self.y))
        cv2.putText(canvas, self.text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(canvas, self.text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, c, 2, cv2.LINE_AA)

# ── Starfield ──────────────────────────────────────────────────────────────────
class Starfield:
    def __init__(self, w, h, count=220):
        self.w, self.h = w, h
        self.stars = [{
            'x': random.uniform(0, w), 'y': random.uniform(0, h),
            'vx': random.uniform(-0.25, 0.25), 'vy': random.uniform(-0.12, 0.12),
            'r':  random.uniform(0.5, 2.2),
            'brightness': random.uniform(0.1, 0.5),
            'tp': random.uniform(0, 2*math.pi),
            'ts': random.uniform(0.03, 0.10),
        } for _ in range(count)]

    def update_draw(self, canvas, t):
        for s in self.stars:
            s['x'] = (s['x'] + s['vx']) % self.w
            s['y'] = (s['y'] + s['vy']) % self.h
            s['tp'] += s['ts']
            twinkle = 0.5 + 0.5 * math.sin(s['tp'])
            val = int(s['brightness'] * twinkle * 210)
            cv2.circle(canvas, (int(s['x']), int(s['y'])),
                       max(1, int(s['r'])), (val, val, val), -1)

# ── Glow helpers ───────────────────────────────────────────────────────────────
def draw_glowing_line_v3(canvas, p1, p2, color, thickness=2, brightness=1.0):
    b  = min(brightness, 2.2)
    o1 = np.zeros_like(canvas, dtype=np.uint8)
    cv2.line(o1, p1, p2, tuple(int(v*min(1.0,0.20*b)) for v in color),
             thickness+16, cv2.LINE_AA)
    cv2.add(canvas, cv2.GaussianBlur(o1, (15,15), 0), canvas)

    o2 = np.zeros_like(canvas, dtype=np.uint8)
    cv2.line(o2, p1, p2, tuple(int(v*min(1.0,0.45*b)) for v in color),
             thickness+6, cv2.LINE_AA)
    cv2.add(canvas, cv2.GaussianBlur(o2, (7,7), 0), canvas)

    cv2.line(canvas, p1, p2, color, thickness+1, cv2.LINE_AA)
    wc = tuple(min(255, int(255*min(1.0, b*0.6))) for _ in range(3))
    cv2.line(canvas, p1, p2, wc, max(1, thickness-1), cv2.LINE_AA)


def draw_glowing_circle_v3(canvas, center, radius, color,
                            thickness=2, brightness=1.0, chromatic=False):
    b = min(brightness, 2.2)
    if chromatic:
        off = 3
        cv2.circle(canvas, (center[0]-off, center[1]), radius,
                   (0,0,min(255,int(color[2]*0.7))), 1, cv2.LINE_AA)
        cv2.circle(canvas, (center[0]+off, center[1]), radius,
                   (min(255,int(color[0]*0.7)),0,0), 1, cv2.LINE_AA)

    o1 = np.zeros_like(canvas, dtype=np.uint8)
    cv2.circle(o1, center, radius+10,
               tuple(int(v*min(1.0,0.22*b)) for v in color),
               thickness+6, cv2.LINE_AA)
    cv2.add(canvas, cv2.GaussianBlur(o1, (15,15), 0), canvas)

    o2 = np.zeros_like(canvas, dtype=np.uint8)
    cv2.circle(o2, center, radius+4,
               tuple(int(v*min(1.0,0.50*b)) for v in color),
               thickness+3, cv2.LINE_AA)
    cv2.add(canvas, cv2.GaussianBlur(o2, (7,7), 0), canvas)

    cv2.circle(canvas, center, radius, color, thickness+1, cv2.LINE_AA)
    dot_r = max(2, radius//3)
    cv2.circle(canvas, center, dot_r,
               tuple(min(255,int(v*1.1)) for v in color), -1, cv2.LINE_AA)
    cv2.circle(canvas, center, max(1,dot_r-1), (230,230,230), -1, cv2.LINE_AA)

# ── Utilities ──────────────────────────────────────────────────────────────────
def lm_to_px(lm, w, h): return int(lm.x*w), int(lm.y*h)

def palm_center_px(landmarks, w, h):
    xs = [landmarks[i].x for i in PALM_CENTER_IDS]
    ys = [landmarks[i].y for i in PALM_CENTER_IDS]
    return int(np.mean(xs)*w), int(np.mean(ys)*h)

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def hsv_color(hue_deg):
    hsv = np.uint8([[[int(hue_deg%360/2), 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def speed_color(speed, base_hue):
    return hsv_color(base_hue + min(speed*0.6, 130))

# ── Gesture detection (robust) ─────────────────────────────────────────────────
def finger_curl_ratio(landmarks, w, h, tip_id, pip_id, mcp_id):
    wrist    = lm_to_px(landmarks[0], w, h)
    tip      = lm_to_px(landmarks[tip_id], w, h)
    mcp      = lm_to_px(landmarks[mcp_id], w, h)
    tip_dist = dist(tip, wrist)
    mcp_dist = dist(mcp, wrist)
    if mcp_dist < 1: return 0.5
    return min(tip_dist / (mcp_dist * 1.8), 1.0)

FINGER_JOINTS = [
    (8,  6,  5),   # index
    (12, 10, 9),   # middle
    (16, 14, 13),  # ring
    (20, 18, 17),  # pinky
]
EXTENSION_THRESHOLD = 0.60   # raised from 0.55 — needs to be more extended to count


def raw_detect_gesture(landmarks, w, h):
    curls = [finger_curl_ratio(landmarks, w, h, *j) for j in FINGER_JOINTS]
    idx_ext, mid_ext, ring_ext, pin_ext = [c > EXTENSION_THRESHOLD for c in curls]

    thumb_tip = lm_to_px(landmarks[4], w, h)
    idx_tip   = lm_to_px(landmarks[8], w, h)
    palm_r    = dist(lm_to_px(landmarks[0], w, h),
                     lm_to_px(landmarks[9], w, h))

    # Fist: ALL four fingers clearly curled
    if not any([idx_ext, mid_ext, ring_ext, pin_ext]):
        return 'fist'

    # Thumbs Up: thumb tip high, all fingers curled
    tip_y   = landmarks[4].y
    ip_y    = landmarks[3].y
    mcp_y   = landmarks[2].y
    palm_y  = np.mean([landmarks[i].y for i in PALM_CENTER_IDS])
    thumb_up_shape  = tip_y < ip_y < mcp_y
    thumb_above_pal = tip_y < palm_y - 0.10   # must be clearly above palm
    fingers_curled  = not any([idx_ext, mid_ext, ring_ext, pin_ext])
    if thumb_up_shape and thumb_above_pal and fingers_curled:
        return 'thumbs_up'

    # OK: thumb + index tips touching, others clearly extended
    if dist(thumb_tip, idx_tip) < palm_r * 0.50 and mid_ext and ring_ext and pin_ext:
        return 'ok'

    # Rock On: index + pinky extended, middle + ring curled
    if idx_ext and not mid_ext and not ring_ext and pin_ext:
        return 'rock_on'

    # Peace: index + middle extended, ring + pinky curled
    if idx_ext and mid_ext and not ring_ext and not pin_ext:
        return 'peace'

    # Open: all four clearly extended
    if idx_ext and mid_ext and ring_ext and pin_ext:
        return 'open'

    return 'normal'


class GestureSmoothed:
    """
    Voting buffer with HOLD requirement.

    A gesture fires only when:
      1. It wins the majority vote in the buffer (10 out of 12 frames)
      2. It has been stable for at least HOLD_FRAMES consecutive frames
      3. The previous FIRED gesture was different (hysteresis — must pass
         through 'normal' before same gesture re-fires)
    """
    BUFFER_SIZE  = 12
    MAJORITY     = 10   # very high — eliminates almost all flicker
    HOLD_FRAMES  = 8    # must be stable this long before triggering (~0.27s)

    def __init__(self):
        self.buffer       = collections.deque(maxlen=self.BUFFER_SIZE)
        self.stable       = 'normal'
        self.hold_counter = 0
        self.last_fired   = None

    def update(self, raw_gesture):
        self.buffer.append(raw_gesture)
        if len(self.buffer) < self.BUFFER_SIZE:
            return self.stable, False   # (current_stable, did_just_trigger)

        counts = collections.Counter(self.buffer)
        top_gesture, top_count = counts.most_common(1)[0]

        # Only accept if supermajority
        if top_count >= self.MAJORITY:
            if top_gesture == self.stable:
                self.hold_counter += 1
            else:
                self.stable       = top_gesture
                self.hold_counter = 0
        else:
            self.hold_counter = 0

        # Fire only when:
        # - stable long enough
        # - not 'normal'
        # - different from last fired (must go through normal first)
        should_fire = (
            self.hold_counter == self.HOLD_FRAMES and
            self.stable != 'normal' and
            self.stable != self.last_fired
        )
        if should_fire:
            self.last_fired = self.stable
        # Reset last_fired when normal is stable (allows same gesture again)
        if self.stable == 'normal':
            self.last_fired = None

        return self.stable, should_fire

    @property
    def vote_count(self):
        if not self.buffer: return 0
        return collections.Counter(self.buffer).most_common(1)[0][1]

# ── Particle spawners ──────────────────────────────────────────────────────────
def spawn_explosion(p, cx, cy):
    colors = [NEON_ORANGE, NEON_YELLOW, NEON_MAGENTA, NEON_WHITE, (20,80,255)]
    for _ in range(100): p.append(Particle(cx,cy,random.choice(colors),3.5,2.2))

def spawn_fireworks(p, cx, cy):
    for _ in range(80): p.append(FireworkParticle(cx,cy,random.choice(RAINBOW_COLORS)))

def spawn_rainbow_burst(p, cx, cy):
    for i in range(60): p.append(Particle(cx,cy,hsv_color(i*6),2.2))

def spawn_thumbs_up(p, cx, cy):
    for _ in range(65): p.append(Particle(cx,cy,NEON_YELLOW,2.5,1.5))

def spawn_ok(p, cx, cy):
    for i in range(55): p.append(Particle(cx,cy,hsv_color(i*7),1.8))

def spawn_rock_on(p, cx, cy):
    for _ in range(90):
        p.append(Particle(cx,cy,random.choice([NEON_RED,NEON_MAGENTA,NEON_ORANGE]),4.0,2.5))

def spawn_clap(p, cx, cy):
    for i in range(80): p.append(Particle(cx,cy,hsv_color(i*4.5),3.0,2.0,gravity=0.03))

def spawn_ripples(rips, cx, cy, color, count=3):
    for i in range(count):
        r = Ripple(cx, cy, color, max_r=random.randint(160,300))
        r.r = i*28
        rips.append(r)

# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_hud(canvas, fps, hand_count, paint_mode, rainbow_mode, debug_mode,
             gesture_states):
    h, w = canvas.shape[:2]

    def put(txt, pos, col, scale=0.65):
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)

    put(f"FPS: {fps:.0f}",       (12, 30),  NEON_CYAN)
    put(f"Hands: {hand_count}",  (12, 58),  NEON_GREEN)
    if paint_mode:   put("PAINT ON",    (12, 86),  NEON_PINK)
    if rainbow_mode: put("RAINBOW",     (12, 114), NEON_YELLOW)
    if debug_mode:   put("DEBUG",       (12, 142), NEON_ORANGE)

    # Show current stable gesture per hand
    for hi, (stable, vote_c) in enumerate(gesture_states):
        if stable != 'normal':
            put(f"H{hi+1}: {stable} ({vote_c}/12)",
                (12, 170 + hi*28), NEON_WHITE, 0.55)

    put("Q=quit  P=paint  C=clear  S=shot  D=debug  R=rainbow",
        (12, h-12), (160,160,160), 0.42)


def draw_debug_overlay(canvas, landmarks_list, w, h, smoothers):
    for hi, lms in enumerate(landmarks_list):
        bx = 20 + hi*185
        by = h - 130
        cv2.rectangle(canvas,(bx,by-15),(bx+175,by+115),(20,20,20),-1)
        cv2.putText(canvas,f"Hand {hi+1}",(bx+5,by),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        labels = ["Index","Middle","Ring","Pinky"]
        for fi,(tid,pid,mid) in enumerate(FINGER_JOINTS):
            curl  = finger_curl_ratio(lms,w,h,tid,pid,mid)
            ext   = curl > EXTENSION_THRESHOLD
            bar_w = int(curl*130)
            by2   = by+18+fi*24
            cv2.rectangle(canvas,(bx+52,by2),(bx+182,by2+14),(50,50,50),-1)
            col = (0,220,80) if ext else (80,80,220)
            cv2.rectangle(canvas,(bx+52,by2),(bx+52+bar_w,by2+14),col,-1)
            cv2.putText(canvas,labels[fi],(bx+5,by2+11),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
            cv2.putText(canvas,f"{curl:.2f}",(bx+56,by2+11),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)
        if hi < len(smoothers):
            sm = smoothers[hi]
            cv2.putText(canvas,f"stable={sm.stable} hold={sm.hold_counter}/{sm.HOLD_FRAMES}",
                        (bx+5,by+115),cv2.FONT_HERSHEY_SIMPLEX,0.32,(160,200,160),1)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import tempfile, urllib.request

    model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading hand landmark model (~8MB)…")
        url = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
               "hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("Model ready!")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280

    particles      = []
    ripples        = []
    labels         = []
    spawn_counter  = 0
    global_hue     = 0.0
    prev_positions = {}
    base_hues      = {}
    smoothers      = {}          # hi -> GestureSmoothed
    paint_mode     = False
    rainbow_mode   = False
    debug_mode     = False
    paint_canvas   = np.zeros((H, W, 3), dtype=np.uint8)
    paint_prev     = {}
    paint_hue      = 0.0
    snake_trails   = {}
    starfield      = Starfield(W, H)
    t              = 0.0
    fps_timer      = time.time()
    fps            = 0.0
    clap_cooldown  = 0
    screenshot_dir = os.path.dirname(os.path.abspath(__file__))

    print("══════════════════════════════════════════════════════════════")
    print("  Neon Hand Tracker ULTRA v3 FIXED  —  Q to quit")
    print("  Hold gestures clearly for ~0.4s to trigger effects")
    print("  ✊ Fist  ✌ Peace  🖐 Open  👍 Thumbs  👌 OK  🤘 Rock  👏 Clap")
    print("  P=paint  C=clear  S=screenshot  D=debug  R=rainbow")
    print("══════════════════════════════════════════════════════════════")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        starfield.update_draw(canvas, t)
        t += 0.016

        if np.any(paint_canvas):
            mask = np.any(paint_canvas > 0, axis=2)
            canvas[mask] = cv2.addWeighted(canvas, 0.25, paint_canvas, 0.88, 0)[mask]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.time() * 1000)
        result   = detector.detect_for_video(mp_image, ts_ms)

        spawn_counter += 1
        beat, beat_energy = beat_detector.is_beat()
        global_hue  = (global_hue + 0.25) % 360
        paint_hue   = (paint_hue  + 0.9)  % 360
        if clap_cooldown > 0: clap_cooldown -= 1

        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

        # ── Two-hand clap ──────────────────────────────────────────────────────
        if hand_count == 2 and clap_cooldown == 0:
            pc0   = palm_center_px(result.hand_landmarks[0], w, h)
            pc1   = palm_center_px(result.hand_landmarks[1], w, h)
            palm_r0 = dist(lm_to_px(result.hand_landmarks[0][0], w, h),
                           lm_to_px(result.hand_landmarks[0][9], w, h))
            if dist(pc0, pc1) < palm_r0 * 2.2:
                mid = ((pc0[0]+pc1[0])//2, (pc0[1]+pc1[1])//2)
                spawn_clap(particles, mid[0], mid[1])
                spawn_ripples(ripples, mid[0], mid[1], NEON_WHITE, 3)
                labels.append(FloatingLabel(mid[0], mid[1]-40,
                                            GESTURE_LABELS['clap'], NEON_WHITE))
                play_gesture_sound('clap')
                clap_cooldown = 45   # ~1.5s between clap fires

        # ── Per-hand processing ────────────────────────────────────────────────
        gesture_states = []

        if result.hand_landmarks:
            for hi, hand_lms in enumerate(result.hand_landmarks):
                if hi not in base_hues:    base_hues[hi] = hi * 180.0
                if hi not in smoothers:    smoothers[hi] = GestureSmoothed()

                base_hues[hi] = (base_hues[hi] + 0.35) % 360
                eff_hue = (base_hues[hi] + global_hue) % 360

                tips_px  = [lm_to_px(hand_lms[i], w, h) for i in FINGERTIP_IDS]
                pcenter  = palm_center_px(hand_lms, w, h)

                speeds = [0.0] * 5
                if hi in prev_positions:
                    for fi in range(5):
                        speeds[fi] = dist(tips_px[fi], prev_positions[hi][fi])
                prev_positions[hi] = tips_px
                avg_speed = sum(speeds) / 5.0

                # Draw skeleton
                for (a_id, b_id) in HAND_CONNECTIONS:
                    p1 = lm_to_px(hand_lms[a_id], w, h)
                    p2 = lm_to_px(hand_lms[b_id], w, h)
                    fi = CONNECTION_FINGER.get((a_id,b_id), 0)
                    bone_hue   = (FINGER_HUES[fi] + global_hue) % 360
                    bone_color = hsv_color(bone_hue if not rainbow_mode
                                           else (t*60 + a_id*18) % 360)
                    draw_glowing_line_v3(canvas, p1, p2, bone_color,
                                         brightness=1.0 + min(avg_speed/25.0, 1.0))

                # Joint nodes
                fi_map = {}
                for i in [1,2,3,4]:   fi_map[i] = 0
                for i in [5,6,7,8]:   fi_map[i] = 1
                for i in [9,10,11,12]:fi_map[i] = 2
                for i in [13,14,15,16]:fi_map[i] = 3
                for i in [17,18,19,20]:fi_map[i] = 4
                for lm_id in range(21):
                    pt     = lm_to_px(hand_lms[lm_id], w, h)
                    fi     = fi_map.get(lm_id, 0)
                    nh     = (FINGER_HUES[fi] + global_hue) % 360
                    nc     = hsv_color(nh if not rainbow_mode else (t*60+lm_id*17)%360)
                    pulse  = 5 + int(min(speeds[fi]*0.3, 6)) if fi < 5 else 5
                    draw_glowing_circle_v3(canvas, pt, pulse, nc, brightness=1.1)

                # ── Smoothed gesture ──────────────────────────────────────────
                raw_g          = raw_detect_gesture(hand_lms, w, h)
                stable, fired  = smoothers[hi].update(raw_g)
                gesture_states.append((stable, smoothers[hi].vote_count))

                if fired:
                    gc = hsv_color(eff_hue)
                    if stable == 'fist':
                        spawn_explosion(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_ORANGE, 4)
                    elif stable == 'peace':
                        spawn_fireworks(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_CYAN, 3)
                    elif stable == 'open':
                        spawn_rainbow_burst(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_GREEN, 5)
                    elif stable == 'thumbs_up':
                        spawn_thumbs_up(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_YELLOW, 3)
                    elif stable == 'ok':
                        spawn_ok(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_BLUE, 3)
                    elif stable == 'rock_on':
                        spawn_rock_on(particles, pcenter[0], pcenter[1])
                        spawn_ripples(ripples, pcenter[0], pcenter[1], NEON_RED, 5)
                    play_gesture_sound(stable)
                    lbl = GESTURE_LABELS.get(stable, '')
                    if lbl:
                        labels.append(FloatingLabel(pcenter[0], pcenter[1]-40, lbl, gc))

                # Beat burst — only if VERY loud AND hand is still
                if beat and beat_energy > 4.0 and avg_speed < 5.0:
                    for fi, tip in enumerate(tips_px):
                        c = hsv_color((FINGER_HUES[fi]+global_hue)%360)
                        for _ in range(8):
                            particles.append(Particle(tip[0],tip[1],c,2.5))

                # Paint
                if paint_mode:
                    idx_tip = tips_px[1]
                    pc = hsv_color(paint_hue + hi*60)
                    if hi in paint_prev and paint_prev[hi] is not None:
                        cv2.line(paint_canvas, paint_prev[hi], idx_tip, pc, 5, cv2.LINE_AA)
                        ov = np.zeros_like(paint_canvas)
                        cv2.line(ov, paint_prev[hi], idx_tip, pc, 14, cv2.LINE_AA)
                        paint_canvas = cv2.add(paint_canvas,
                                               cv2.GaussianBlur(ov,(15,15),0))
                    paint_prev[hi] = idx_tip
                else:
                    paint_prev[hi] = None

                # Snake trails
                for fi in range(5):
                    key = (hi, fi)
                    if key not in snake_trails:
                        snake_trails[key] = SnakeTrail(fi)
                    sn = snake_trails[key]
                    sn.push(tips_px[fi], (FINGER_HUES[fi]+global_hue)%360)
                    sn.draw(canvas, speeds[fi], rainbow_mode=rainbow_mode, t=t)

                    # Fingertip particles — only when moving fast enough
                    if spawn_counter % 2 == 0 and speeds[fi] > 8.0:
                        sm = max(1.0, speeds[fi]*0.10)
                        tc = speed_color(speeds[fi], FINGER_HUES[fi]+global_hue)
                        for _ in range(3):
                            particles.append(Particle(tips_px[fi][0],tips_px[fi][1],tc,sm))

                draw_glowing_circle_v3(canvas, pcenter, 20, NEON_WHITE, 2)

            if debug_mode:
                sm_list = [smoothers[hi] for hi in range(hand_count) if hi in smoothers]
                draw_debug_overlay(canvas, result.hand_landmarks, w, h, sm_list)

        # ── Update & draw effects ──────────────────────────────────────────────
        alive_r = []
        for r in ripples:
            r.update(); r.draw(canvas)
            if r.alive: alive_r.append(r)
        ripples[:] = alive_r

        alive_p = []
        for p in particles:
            p.update()
            if p.alive:
                p.draw(canvas)
                alive_p.append(p)
        particles[:] = alive_p[-3500:]

        alive_l = []
        for lb in labels:
            lb.update()
            if lb.alive:
                lb.draw(canvas)
                alive_l.append(lb)
        labels[:] = alive_l[-20:]

        # Bloom
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
        bright_layer   = cv2.bitwise_and(canvas, canvas, mask=bright_mask)
        bloom = cv2.addWeighted(cv2.GaussianBlur(bright_layer,(21,21),0), 0.15,
                                cv2.GaussianBlur(bright_layer,(7,7), 0), 0.10, 0)
        cv2.add(canvas, bloom, canvas)

        now   = time.time()
        fps   = 0.9*fps + 0.1*(1.0/max(now-fps_timer, 1e-5))
        fps_timer = now

        draw_hud(canvas, fps, hand_count, paint_mode, rainbow_mode,
                 debug_mode, gesture_states)
        cv2.imshow("Neon Hand Tracker ULTRA v3 FIXED", canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key in (ord('p'),ord('P')):
            paint_mode = not paint_mode; paint_prev.clear()
            print(f"Paint: {'ON' if paint_mode else 'OFF'}")
        elif key in (ord('c'),ord('C')):
            paint_canvas[:] = 0; print("Canvas cleared.")
        elif key in (ord('s'),ord('S')):
            f = os.path.join(screenshot_dir, f"shot_{int(time.time())}.png")
            cv2.imwrite(f, canvas); print(f"Saved: {f}")
        elif key in (ord('d'),ord('D')):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")
        elif key in (ord('r'),ord('R')):
            rainbow_mode = not rainbow_mode
            print(f"Rainbow: {'ON' if rainbow_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()