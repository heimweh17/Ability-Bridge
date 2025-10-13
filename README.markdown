# Ability Bridge

**Hands-Free Computer Control through Facial Gestures and AI Vision**

Ability-Bridge is an assistive technology project designed to enable computer interaction without traditional input devices. By combining real-time facial landmark tracking and intelligent gesture interpretation, it allows users to control a cursor, perform clicks, and type text using only head movement, mouth gestures, and eyebrow raises.

## Overview

Many individuals with limited mobility face barriers when using a keyboard or mouse. Ability-Bridge transforms a standard webcam into a smart, camera-based interface that responds to facial movements. It bridges human ability and digital accessibility using algorithms from computer vision, geometry, and signal processing.

## Core Features

| Feature | Description |
| --- | --- |
| **Head-Pose Cursor Control** | Computes yaw and pitch angles from facial landmarks to translate head movement into 2D cursor motion. |
| **Mouth-Based Morse Typing** | Measures the mouth aspect ratio (MAR) to distinguish short and long openings, generating Morse code that converts into text. |
| **Eyebrow-Raise Clicks** | Detects eyebrow elevation relative to neutral position to simulate mouse clicks without blinks. |
| **Adaptive Calibration** | Automatically calibrates neutral and active gesture thresholds for reliable operation across users and lighting conditions. |
| **Real-Time Processing** | Achieves smooth performance at typical webcam frame rates using MediaPipe FaceMesh and OpenCV optimizations. |

## Technical Architecture

### Technologies and Libraries

- **Python 3**
- **OpenCV**: Video stream processing and geometric computations
- **MediaPipe FaceMesh**: Facial landmark detection and pose estimation
- **NumPy**: Vectorized math operations
- **PyAutoGUI**: Keyboard and mouse event synthesis

### Pipeline Summary

1. Capture webcam frames and extract 468 facial landmarks with MediaPipe.
2. Compute key ratios and angles:
   - Mouth aspect ratio (MAR) → Morse dot/dash detection
   - Brow-eye distance → click trigger
   - Head yaw/pitch → cursor movement
3. Apply temporal smoothing and dead-zone filtering for steady control.
4. Execute corresponding keyboard or mouse events using PyAutoGUI.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Alex-Liu/Ability-Bridge.git
   cd Ability-Bridge
   ```

2. **Install dependencies**:

   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

3. **Run the program**:

   ```bash
   python AbilityBridge.py
   ```

4. **During startup, follow the on-screen calibration steps**:

   - Keep your mouth closed and open when prompted.
   - Relax eyebrows for neutral capture.
   - The system will determine individual gesture thresholds.

5. **Once active**:

   - Turn your head to move the cursor.
   - Open mouth briefly for a “dot” or longer for a “dash.”
   - Pause to commit a letter or raise eyebrows to click.

## Example Applications

- Accessibility tools for individuals with spinal cord injuries or neuromuscular disorders
- Research on non-verbal human–computer interaction
- Educational demonstrations of real-time facial landmark analysis
- Prototype foundation for contactless interfaces or AR/VR environments

## Design Highlights

- Implements 3D pose estimation using camera intrinsic matrices and solvePnP.
- Uses temporal smoothing (exponential filter) for stable cursor motion.
- Employs hysteresis thresholds for robust open/close gesture detection.
- Achieves hardware-independent performance using only a webcam and Python libraries.

## Future Directions

- Integrate speech synthesis for Morse-to-speech output.
- Extend gesture recognition to include eye movement or tongue gestures.
- Add a graphical UI for calibration and configuration.
- Package as a cross-platform desktop application with accessibility settings.

## Author and Purpose

Developed by **Alex Liu** as a demonstration of applied computer vision and assistive technology engineering. Ability Bridge reflects a commitment to accessibility, inclusivity, and human-centered software design — showing how algorithmic thinking and empathy can combine to expand digital reach.
