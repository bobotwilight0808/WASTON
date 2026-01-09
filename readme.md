# Waston

Waston is an on-device detection demo that includes:

- A **Raspberry Pi** desktop GUI (PyQt5 + OpenCV + ONNX Runtime + Picamera2)
- An **Android** NCNN-based demo app for running lightweight YOLO-style models

This repository contains two independent modules under one workspace.

## Repository layout

- `respberryPi/` — Raspberry Pi GUI app (`qt.py`) and ONNX model (`best.onnx`)
- `android/` — Android Studio project (NCNN + OpenCV)

## Raspberry Pi (PyQt + ONNX Runtime)

### Requirements

- Python 3.10
- A Raspberry Pi environment with camera support (the app uses `picamera2`)
- An X11/desktop session (the script sets `DISPLAY=:0`)

### Install

From the repository root:

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
cd respberryPi
python qt.py
```

In the UI, you can start the camera, capture an image, load an image from disk, and toggle detection.

## Android (NCNN)

The Android module is a sample NCNN + OpenCV project for YOLOv5-Lite style object detection.

### Build prerequisites

- Android Studio
- Android NDK + CMake
- Prebuilt or locally built NCNN and OpenCV-Mobile packages

### Build steps (summary)

1. Get NCNN for Android (from releases or build it yourself):
	- https://github.com/Tencent/ncnn/releases
	- Extract into `android/app/src/main/jni/`
	- Update the `ncnn_DIR` path in `android/app/src/main/jni/CMakeLists.txt`
2. Get OpenCV-Mobile:
	- https://github.com/nihui/opencv-mobile
	- Extract into `android/app/src/main/jni/`
	- Update the `OpenCV_DIR` path in `android/app/src/main/jni/CMakeLists.txt`
3. Ensure model assets (`*.param` and `*.bin`) are present in:
	- `android/app/src/main/assets/`
4. Open `android/` in Android Studio and build/run.

For more details, see `android/README.md`.

## Credits / Based on

The Android module is adapted from, and/or references the following upstream projects:

- NCNN: https://github.com/Tencent/ncnn
- OpenCV-Mobile: https://github.com/nihui/opencv-mobile
- ncnn-android-yolov5 (reference implementation): https://github.com/nihui/ncnn-android-yolov5
- ncnn-android-yolox (reference): https://github.com/FeiGeChuanShu/ncnn-android-yolox
- YOLOv5-Lite: https://github.com/ppogg/YOLOv5-Lite
- Model zoo assets referenced by the Android demo:
  https://github.com/ppogg/ncnn-android-v5lite/tree/master/app/src/main/assets

The Raspberry Pi module uses ONNX Runtime, OpenCV, and PyQt5.