# OpenCV-FacialBlur

Real-time facial detection and blurring from a webcam feed.

Uses **MediaPipe Face Detection** (default) or OpenCV's classic Haar cascade as the detector backend, and applies a Gaussian blur over every detected face.

![Blur Webcam](https://user-images.githubusercontent.com/49893887/181924521-afa99c39-d37a-4e81-8c89-086e8c90e153.png)

---

## Requirements

- Python 3.11+
- A webcam

## Setup

```bash
# clone
git clone https://github.com/<you>/OpenCV-FacialBlur.git
cd OpenCV-FacialBlur

# install deps (pick one)
pip install -r requirements.txt
# or with pipenv
pipenv install
```

## Usage

```bash
python blur.py [options]

Options:
  --camera INT      Camera device index (default: 0)
  --blur INT        Gaussian blur kernel size, must be odd (default: 51)
  --detector STR    Face detector backend: mediapipe | haar (default: mediapipe)
```

**Examples**

```bash
# default — MediaPipe detector, camera 0, blur strength 51
python blur.py

# use the Haar cascade backend on a second camera
python blur.py --detector haar --camera 1

# stronger blur
python blur.py --blur 99
```

Press **q** in the output window to quit.

---

## How it works

1. Each frame is fed to the selected face detector.
2. Detected face bounding boxes are blurred in-place with `cv2.GaussianBlur`.
3. The result is displayed in a window in real time.

**MediaPipe** (`--detector mediapipe`) is the default — it uses a lightweight CNN and handles faces at angles far better than the Haar cascade.

**Haar cascade** (`--detector haar`) uses OpenCV's bundled `haarcascade_frontalface_default.xml`. It's faster on very low-end hardware but produces more false positives and struggles with non-frontal faces.
