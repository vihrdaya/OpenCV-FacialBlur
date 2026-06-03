import argparse
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time facial blur via webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--blur", type=int, default=51, help="Gaussian blur kernel size, must be odd (default: 51)")
    parser.add_argument(
        "--detector",
        choices=["mediapipe", "haar"],
        default="mediapipe",
        help="Face detector backend (default: mediapipe)",
    )
    return parser.parse_args()


def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def blur_region(frame: np.ndarray, x: int, y: int, w: int, h: int, ksize: int) -> None:
    roi = frame[y : y + h, x : x + w]
    frame[y : y + h, x : x + w] = cv.GaussianBlur(roi, (ksize, ksize), 0)


def detect_faces_mediapipe(frame: np.ndarray, detector) -> list[tuple[int, int, int, int]]:
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = detector.process(rgb)
    boxes: list[tuple[int, int, int, int]] = []
    if not results.detections:
        return boxes
    h, w = frame.shape[:2]
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)
        boxes.append((x, y, bw, bh))
    return boxes


def detect_faces_haar(frame: np.ndarray, cascade: cv.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def main() -> None:
    args = parse_args()
    ksize = ensure_odd(args.blur)

    # Open webcam
    video = cv.VideoCapture(args.camera)
    if not video.isOpened():
        sys.exit(f"Error: could not open camera {args.camera}.")

    # Set up detector
    mp_detector = None
    haar_cascade = None
    if args.detector == "mediapipe":
        mp_face = mp.solutions.face_detection
        mp_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    else:
        haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        if haar_cascade.empty():
            sys.exit("Error: could not load Haar cascade file.")

    print(f"Running facial blur  |  detector={args.detector}  camera={args.camera}  blur={ksize}")
    print("Press  q  to quit.")

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Warning: failed to read frame, retrying...")
                continue

            if args.detector == "mediapipe":
                faces = detect_faces_mediapipe(frame, mp_detector)
            else:
                faces = detect_faces_haar(frame, haar_cascade)

            for x, y, w, h in faces:
                blur_region(frame, x, y, w, h, ksize)

            cv.imshow("Facial Blur  (q to quit)", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv.destroyAllWindows()
        if mp_detector:
            mp_detector.close()


if __name__ == "__main__":
    main()
