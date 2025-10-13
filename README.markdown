# **Ability Bridge**: Hands-Free Computer Control

FacialHelm is an assistive technology project that transforms a standard webcam into a comprehensive, hands-free interface for computer control. This application allows users to type using Morse code with their mouth, navigate the cursor with their head, and click by blinking.

## The Vision: Creativity Meets Accessibility

The digital world should be accessible to everyone. This project was born from a creative challenge: How can we empower individuals with motor impairments to seamlessly interact with a computer using only the hardware they already own? Instead of relying on expensive, specialized equipment, FacialHelm is a software-only solution that turns a user's own facial expressions into a powerful command center. It's a testament to the idea that thoughtful software can create a more inclusive and helpful world.

## Core Features

- 🖱️ **Intuitive Mouse Navigation**: Control the cursor's movement in real-time by simply turning your head. The motion is smoothed to ensure a stable and precise pointing experience.
- ✍️ **Mouth-to-Morse Typing**: A novel typing system that translates mouth gestures into Morse code. A short mouth opening is a 'dot' (·) and a long opening is a 'dash' (—), enabling access to the full keyboard.
- 👆 **Blink-Powered Clicks**: Perform a left mouse click with a deliberate blink, allowing for effortless interaction with buttons, links, and applications.
- ⚙️ **Smart Auto-Calibration**: To ensure the application is helpful and easy to use, a one-time calibration routine runs on startup. It learns the user's unique facial structure to set personalized thresholds for mouth movements.

## Technical Deep Dive & Learning Journey

This project was a fantastic opportunity to learn and apply advanced concepts in computer vision and human-computer interaction.

### Tech Stack

- **Python**: Chosen for its rich scientific computing ecosystem and rapid development capabilities.
- **OpenCV**: Leveraged for robust and high-performance real-time video capture and processing.
- **MediaPipe**: Utilized for its state-of-the-art facial landmark detection models, which provide the high-accuracy data needed for precise gesture recognition without requiring a powerful GPU.
- **PyAutoGUI**: Employed to bridge the gap between gesture detection and system action, programmatically controlling the OS-level mouse and keyboard.

### Challenges & Solutions

Building an intuitive and reliable facial gesture controller involved solving several interesting problems, demonstrating a strong capacity for learning and creative problem-solving:

1. **Challenge**: Initial head-tracking was jittery due to minor, natural head movements.

   - **Solution**: Implemented an exponential moving average (EMA) smoothing algorithm. This filtered out high-frequency noise, resulting in a stable and predictable cursor that felt much more professional and less fatiguing to use.

2. **Challenge**: Distinguishing an intentional, command-driven blink from a natural, involuntary one.

   - **Solution**: Moved beyond a simple threshold. The system now requires the user's Eye Aspect Ratio (EAR) to remain below the threshold for a minimum number of consecutive frames. This small change drastically reduced false positives and made the click action reliable and intentional.

3. **Challenge**: A "one-size-fits-all" threshold for mouth opening wasn't effective across different users.

   - **Solution**: Designed and implemented an automatic calibration sequence. This user-centric feature learns the individual's unique range of motion, making the Morse code function significantly more accurate and accessible from the very first use.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/FacialHelm.git
   cd FacialHelm
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python main.py
   ```

   Follow the on-screen instructions to calibrate, and then take control with your face!

## Future Goals

This project has a lot of potential for growth. Future plans include:

- Adding more gestures, like an eyebrow raise for right-clicking.
- Developing a simple GUI for easier configuration of sensitivity and settings.
- Implementing a "dwell-to-click" feature as an alternative to blinking.
