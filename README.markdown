# Ability-Bridge
**Hands-Free Computer Control with Facial Gestures and Computer Vision**

Ability-Bridge is an assistive-technology project that enables people with limited mobility to control a computer without a keyboard or mouse. Using only a standard webcam, the system interprets facial gestures in real time to move the mouse cursor, click, delete, press Enter, and type text via Morse code.

**Mission:** lower the barrier to digital access by transforming small, reliable facial gestures into precise input events.

## What Ability-Bridge does
* **Head-pose cursor control** — translate yaw/pitch from 3D face pose into smooth 2D cursor motion with dead-zones and velocity smoothing.
* **Mouth-based Morse typing** — classify short/long mouth openings (MAR ratio) as dot/dash, decode letters, and auto-insert word spaces with robust gap logic.
* **Eyebrow-raise click** — normalize brow-to-eye distance by eye width and trigger a single mouse click on a rising gesture with hysteresis + debounce (no accidental blink clicks).
* **Long-blink commands** — eye closure duration maps to discrete actions:
  * 3–6 s → Backspace
  * 6–9 s → Enter

## Why it matters
Traditional input devices can be a barrier for people with spinal cord injuries, neuromuscular conditions, or temporary mobility constraints. Ability-Bridge demonstrates a camera-only, low-cost pathway to independence and participation—no special hardware beyond a webcam.

## Results (typical on a mid-range laptop)
* 30 FPS at 640×480 with MediaPipe FaceMesh (468 landmarks) and OpenCV.
* < 100 ms end-to-end response for click and cursor updates after calibration.
* Hysteresis + debounce substantially reduce false activations in common lighting conditions.

*(Metrics are observed on a mid-range CPU; your results may vary with camera quality and lighting.)*

## How it works (technical overview)
1. **Landmark extraction:** MediaPipe FaceMesh returns 468 landmarks per frame.
2. **Geometry & ratios**
   * **Head pose:** solvePnP with a sparse facial model → yaw/pitch; convert to dx/dy with dead-zones and gains.
   * **Mouth aspect ratio (MAR):** vertical 13–14 ÷ horizontal 61–291 → dot/dash timing.
   * **Brow raise ratio:** vertical (brow-up − eye-upper) ÷ eye width, averaged per side → scale-invariant click signal.
   * **EAR (eye aspect ratio):** average of left/right → robust long-blink duration.
3. **Signal processing:** exponential smoothing (velocity), hysteresis thresholds (open/close), debouncing (click spacing), gap-based tokenization (letter/word).
4. **Synthesis:** PyAutoGUI emits mouse moves, clicks, backspace/enter, and typed characters.

## Install and run
```bash
git clone https://github.com/<your-username>/Ability-Bridge.git
cd Ability-Bridge
pip install opencv-python mediapipe pyautogui numpy
python AbilityBridge.py
```
## Startup flow
1. Follow on-screen calibration:
   * Phase 1: keep mouth closed
   * Phase 2: keep mouth open
   * Phase 3: relax eyebrows (neutral)
2. Begin using gestures (see Controls below).
3. Press `q` in the preview window to exit.

*Deliberately no packaging is included here; this repository emphasizes readable source and ease of modification for accessibility researchers, clinicians, and contributors.*

## Controls
| Gesture | Action |
|---------|--------|
| **Move cursor** | Slowly rotate head; motion begins when yaw/pitch exceeds dead-zones. |
| **Click** | Raise eyebrows (single click on the rising edge). |
| **Type** | Open mouth briefly (dot) or longer (dash); pause to commit the letter; a longer pause inserts a space. |
| **Delete** | Close eyes for 3–6 seconds. |
| **Enter** | Close eyes for 6–9 seconds. |
| **Quit** | Press `q` while the preview window is focused. |

## Configuration (key parameters)
You can fine-tune behavior by editing the constants at the top of `AbilityBridge.py`:

| Category | Parameters |
|----------|------------|
| **Cursor & pose** | `YAW_DEAD_DEG`, `PITCH_DEAD_DEG`, `GAIN_X`, `GAIN_Y`, `SMOOTH_ALPHA`, `MAX_STEP`, `INVERT_X`, `INVERT_Y` |
| **Morse typing** | `LETTER_GAP`, `WORD_GAP`, `DOT_MAX`, `DASH_MIN` |
| **Eyebrow click** | `BROW_RAISE_MULT`, `BROW_RELEASE_MULT`, `BROW_MIN_GAP_S` |
| **Long-blink actions** | `BLINK_EAR_THR`, `BLINK_HYST` (classification threshold + hysteresis) and the durations in the code paths (3–6 s backspace, 6–9 s enter) |

**Suggested tuning workflow:**
1. Verify stable MAR and EAR readings in the overlay; adjust thresholds slightly (±0.02) if needed.
2. If clicks are too sensitive, raise `BROW_RAISE_MULT` or increase `BROW_MIN_GAP_S`.
3. If cursor overshoots, increase dead-zones or decrease gains; adjust `SMOOTH_ALPHA` for steadiness.

## Reliability & safety
* **Hysteresis:** distinct raise/release thresholds prevent flicker.
* **Debounce:** minimum spacing between clicks avoids duplicates.
* **Dead-zones:** ignore tiny head jitters.
* **Fail-safe:** PyAutoGUI's corner failsafe can be enabled if desired; ensure you understand how to recover focus if the pointer moves unexpectedly.
* **Lighting:** front-facing, even lighting improves landmark stability—avoid backlit scenes.

## What this project demonstrates (for reviewers and recruiters)
* **Applied computer vision:** landmark geometry, pose estimation, and ratio-based signals.
* **Real-time systems:** frame-rate-constrained processing with smoothing and event timing.
* **Human-computer interaction:** mapping micro-gestures to ergonomic, low-error input.
* **Accessibility engineering:** inclusive design choices, calibration for different users, and low-cost hardware (webcam-only).

## Roadmap
* Configuration UI for threshold tuning and calibration persistence
* On-screen Morse guide and training prompts
* Optional speech synthesis for Morse output
* Multi-gesture profiles (e.g., toggle-drag, double-raise for double-click)
* Cross-platform packaging as a separate release branch (opt-in)
