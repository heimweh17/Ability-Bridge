import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Set to True for consistency
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MOUTH_VERT_TOP = 13
MOUTH_VERT_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
YAW_DEAD_DEG = 45.0
PITCH_DEAD_DEG = 10.0
GAIN_X = 7.0
GAIN_Y = 7.0
SMOOTH_ALPHA = 0.25
MAX_STEP = 80
BLINK_EAR_THR = 0.20
BLINK_HYST = 0.05  # Hysteresis for eye open
BLINK_MIN_FR = 2
INVERT_X = True
INVERT_Y = False
IDX_NOSE = 1
IDX_CHIN = 152
IDX_RE = 263
IDX_LE = 33
IDX_RM = 291
IDX_LM = 61
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [43.3, 32.7, -26.0],
    [-43.3, 32.7, -26.0],
    [28.9, -28.9, -24.1],
    [-28.9, -28.9, -24.1],
], dtype=np.float64)
EYE_L = dict(U=159, D=145, L=33, R=133)
EYE_R = dict(U=386, D=374, L=263, R=362)
# --- Eyebrow-raise click config ---
BROW_RAISE_MULT = 1.18  # ratio above neutral to count as "raised"
BROW_RELEASE_MULT = 1.08  # fall-back threshold to rearm after a raise
BROW_MIN_GAP_S = 0.30  # min seconds between clicks (debounce)
# FaceMesh landmark IDs for brows/eyes
L_BROW_UP = 105
R_BROW_UP = 334
L_EYE = dict(UP=159, L=33, R=133)
R_EYE = dict(UP=386, L=263, R=362)

def _pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float64)

def brow_raise_ratio(lm, w, h):
    """
    Scale-invariant eyebrow height metric:
    avg( (brow_up - eye_upper)_vert_distance / eye_width )
    Larger => brows higher.
    """
    # Left eye
    pL_up = _pt(lm, L_EYE['UP'], w, h)
    pL_l = _pt(lm, L_EYE['L'], w, h)
    pL_r = _pt(lm, L_EYE['R'], w, h)
    pLb_up = _pt(lm, L_BROW_UP, w, h)
    # Right eye
    pR_up = _pt(lm, R_EYE['UP'], w, h)
    pR_l = _pt(lm, R_EYE['L'], w, h)
    pR_r = _pt(lm, R_EYE['R'], w, h)
    pRb_up = _pt(lm, R_BROW_UP, w, h)
    # vertical distances (abs y-diff) and eye widths
    dL = abs(pLb_up[1] - pL_up[1])
    dR = abs(pRb_up[1] - pR_up[1])
    wL = np.linalg.norm(pL_l - pL_r)
    wR = np.linalg.norm(pR_l - pR_r)
    rL = dL / max(wL, 1e-6)
    rR = dR / max(wR, 1e-6)
    return 0.5 * (rL + rR)

def euclid(a, b): return np.linalg.norm(a - b)

def mar_value(lm, W, H):
    """MAR = Vertical(13-14) / Horizontal(61-291), larger means more open"""
    p_top = np.array([lm[MOUTH_VERT_TOP].x * W, lm[MOUTH_VERT_TOP].y * H])
    p_bot = np.array([lm[MOUTH_VERT_BOTTOM].x * W, lm[MOUTH_VERT_BOTTOM].y * H])
    p_l = np.array([lm[MOUTH_LEFT].x * W, lm[MOUTH_LEFT].y * H])
    p_r = np.array([lm[MOUTH_RIGHT].x * W, lm[MOUTH_RIGHT].y * H])
    vert = euclid(p_top, p_bot)
    horiz = euclid(p_l, p_r)
    return (vert / horiz) if horiz > 0 else 0.0

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"
}

LETTER_GAP = 0.9
WORD_GAP = 2.2
DOT_MAX = 0.25
DASH_MIN = 0.30

def calibrate(cap, secs_hold=3.0):
    """
    Guides you to: keep mouth closed -> keep mouth open, each for about 3 seconds.
    Calculates the average MAR for closed/open mouth to generate thresholds with hysteresis (open/close).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    def collect(prompt, keep_open):
        vals = []
        t0 = time.time()
        while time.time() - t0 < secs_hold:
            ok, fr = cap.read()
            if not ok: continue
            H, W = fr.shape[:2]
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                m = mar_value(lm, W, H)
                vals.append(m)
                cv2.putText(fr, f"Calibrating: {prompt}", (10, 30), font, 0.8, (0, 255, 255), 2)
                cv2.putText(fr, f"MAR:{m:.3f}", (10, 60), font, 0.7, (255, 180, 180), 2)
                cv2.putText(fr, f"Remain: {secs_hold - (time.time() - t0):.1f}s", (10, 90), font, 0.7, (200, 200, 200), 2)
            else:
                cv2.putText(fr, "No face detected...", (10, 30), font, 0.8, (0, 0, 255), 2)
            cv2.imshow("Mouth Morse", fr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        return np.median(vals) if vals else None
    print("Calibration Phase 1: Please 'keep MOUTH CLOSED' for about 3 seconds...")
    closed = collect("Keep MOUTH CLOSED", keep_open=False)
    if closed is None: return None
    print("Calibration Phase 2: Please 'keep MOUTH OPEN' for about 3 seconds...")
    opened = collect("Keep MOUTH OPEN", keep_open=True)
    if opened is None: return None
    MAR_OPEN_T = opened * 0.75 + closed * 0.25
    MAR_CLOSE_T = opened * 0.35 + closed * 0.65
    if not (opened > closed):
        opened, closed = max(opened, closed + 0.05), closed
    MAR_OPEN_T = max(MAR_OPEN_T, closed + 0.05)
    MAR_CLOSE_T = min(MAR_CLOSE_T, opened - 0.05)
    print("Calibration Phase 3: Relax your eyebrows (neutral) for ~1.5s...")
    vals = []
    t0 = time.time()
    while time.time() - t0 < 1.5:
        ok, fr = cap.read()
        if not ok: continue
        H, W = fr.shape[:2]
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            vals.append(brow_raise_ratio(lm, W, H))
        cv2.putText(fr, "Keep eyebrows neutral...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Mouth Morse", fr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None
    brow_neutral = np.median(vals) if vals else 0.10  # fallback small number
    print(f"[Calibration] Brow neutral ratio ≈ {brow_neutral:.3f}")
    return (MAR_OPEN_T, MAR_CLOSE_T, brow_neutral)

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pU = np.array([landmarks[eye_idx['U']].x * w, landmarks[eye_idx['U']].y * h])
    pD = np.array([landmarks[eye_idx['D']].x * w, landmarks[eye_idx['D']].y * h])
    pL = np.array([landmarks[eye_idx['L']].x * w, landmarks[eye_idx['L']].y * h])
    pR = np.array([landmarks[eye_idx['R']].x * w, landmarks[eye_idx['R']].y * h])
    v = np.linalg.norm(pU - pD)
    hlen = np.linalg.norm(pL - pR)
    return 0.0 if hlen < 1e-6 else v / hlen

def get_pose_angles(landmarks, w, h):
    pts_2d = np.array([
        [landmarks[IDX_NOSE].x * w, landmarks[IDX_NOSE].y * h],
        [landmarks[IDX_CHIN].x * w, landmarks[IDX_CHIN].y * h],
        [landmarks[IDX_RE].x * w, landmarks[IDX_RE].y * h],
        [landmarks[IDX_LE].x * w, landmarks[IDX_LE].y * h],
        [landmarks[IDX_RM].x * w, landmarks[IDX_RM].y * h],
        [landmarks[IDX_LM].x * w, landmarks[IDX_LM].y * h],
    ], dtype=np.float64)
    focal = w
    cam_mtx = np.array([[focal, 0, w / 2],
                        [0, focal, h / 2],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts_2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    pitch, yaw, roll = angles
    return float(yaw), float(pitch)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    got = calibrate(cap, secs_hold=3.0)
    if got is None:
        print("Calibration failed or was interrupted.")
        cap.release()
        cv2.destroyAllWindows()
        return
    MAR_OPEN_T, MAR_CLOSE_T, BROW_NEUTRAL = got
    mouth_open = False
    open_start_t = None
    cur_morse = ""
    last_event_t = time.time()
    last_char_type = None
    missing = 0
    MISSING_LIMIT = 15
    vx_s, vy_s = 0.0, 0.0
    last_click_time = 0
    brow_is_up = False
    eye_closed = False
    eye_close_start = None
    print("Start: Short mouth open(·), long mouth open(—); Pause to complete a letter; 'q' to quit.")
    print("Eye closure: 3-6s for backspace, 6-9s for enter.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)  # Mirror flip
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        mar = None
        lm = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            mar = mar_value(lm, W, H)
            missing = 0
        else:
            missing += 1
        now = time.time()
        if missing >= MISSING_LIMIT and mouth_open:
            mouth_open = False
            dur = now - (open_start_t or now)
            if dur <= DOT_MAX:
                cur_morse += "."
            elif dur >= DASH_MIN:
                cur_morse += "-"
            open_start_t = None
            last_event_t = now
        if mar is not None:
            if not mouth_open and mar > MAR_OPEN_T:
                mouth_open = True
                open_start_t = now
            elif mouth_open and mar < MAR_CLOSE_T:
                mouth_open = False
                dur = now - (open_start_t or now)
                if dur <= DOT_MAX:
                    cur_morse += "."
                elif dur >= DASH_MIN:
                    cur_morse += "-"
                open_start_t = None
                last_event_t = now
        gap = now - last_event_t
        # --- Commit letter ---
        if cur_morse and not mouth_open and gap > LETTER_GAP:
            ch = MORSE.get(cur_morse, "")
            if ch:
                pyautogui.typewrite(ch.lower())
                last_char_type = 'letter'
            cur_morse = ""
            last_event_t = now
        # --- Add space after letter ---
        if (not mouth_open) and (cur_morse == "") and (gap > WORD_GAP) and (last_char_type == 'letter'):
            pyautogui.typewrite(" ")
            last_char_type = 'space'
            last_event_t = now
        if lm is not None:
            # Eye closure detection for backspace/enter
            ear_l = eye_aspect_ratio(lm, EYE_L, W, H)
            ear_r = eye_aspect_ratio(lm, EYE_R, W, H)
            avg_ear = (ear_l + ear_r) / 2.0
            if not eye_closed and avg_ear < BLINK_EAR_THR:
                eye_closed = True
                eye_close_start = now
            elif eye_closed and avg_ear > (BLINK_EAR_THR + BLINK_HYST):
                eye_closed = False
                if eye_close_start is not None:
                    dur = now - eye_close_start
                    if 3 <= dur < 6:
                        pyautogui.press('backspace')
                    elif 6 <= dur <= 9:
                        pyautogui.press('enter')
                eye_close_start = None
            # Brow raise for click
            br = brow_raise_ratio(lm, W, H)
            raise_thr = BROW_NEUTRAL * BROW_RAISE_MULT
            release_thr = BROW_NEUTRAL * BROW_RELEASE_MULT
            if not brow_is_up and br > raise_thr and (now - last_click_time) > BROW_MIN_GAP_S:
                pyautogui.click()
                last_click_time = now
                brow_is_up = True
            elif brow_is_up and br < release_thr:
                brow_is_up = False
            # Head pose for mouse
            yaw, pitch = get_pose_angles(lm, W, H)
            if yaw is not None:
                over_yaw = max(0.0, abs(yaw) - YAW_DEAD_DEG)
                a = 180 - abs(pitch)
                over_pitch = max(0.0, a - PITCH_DEAD_DEG)
                dx = dy = 0.0
                if over_yaw > 0:
                    sgn = -1.0 if INVERT_X else 1.0
                    dx = sgn * np.sign(yaw) * over_yaw * GAIN_X
                if over_pitch > 0:
                    base = -1.0 if not INVERT_Y else 1.0
                    dy = base * np.sign(pitch) * over_pitch * GAIN_Y
                vx_s = (1 - SMOOTH_ALPHA) * vx_s + SMOOTH_ALPHA * dx
                vy_s = (1 - SMOOTH_ALPHA) * vy_s + SMOOTH_ALPHA * dy
                step_x = int(np.clip(vx_s, -MAX_STEP, MAX_STEP))
                step_y = int(np.clip(vy_s, -MAX_STEP, MAX_STEP))
                if step_x or step_y:
                    try:
                        pyautogui.moveRel(step_x, step_y, duration=0)
                    except:
                        pass
                cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}", (10, 30),
                            font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"dx:{step_x} dy:{step_y}", (10, 60),
                            font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Brow r:{br:.3f} up:{brow_is_up} (raise>{raise_thr:.3f}/rel<{release_thr:.3f})",
                        (10, 90), font, 0.6, (100, 220, 255), 2)
            cv2.putText(frame, f"MAR {mar:.3f} (O{MAR_OPEN_T:.2f}/R{MAR_CLOSE_T:.2f})",
                        (10, 120), font, 0.7, (255, 160, 160), 2)
            cv2.putText(frame, f"EAR:{avg_ear:.3f} closed:{eye_closed}",
                        (10, 150), font, 0.7, (255, 160, 160), 2)
        else:
            cv2.putText(frame, "No face", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Morse: {cur_morse}", (10, 180), font, 0.8, (0, 200, 0), 2)
        cv2.putText(frame, "Mouth Morse | q=Quit", (10, H - 14), font, 0.6, (200, 200, 200), 1)
        cv2.imshow("Mouth Morse", frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
