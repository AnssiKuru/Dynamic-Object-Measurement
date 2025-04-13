# Dynamic Object Measurement

This project automatically detects objects from a reference image and allows real-time distance and height measurement using YOLOv4-tiny and OpenCV.

## Features
- Dynamic object detection (best match selected automatically)
- Real-time object measurement
- Works with any detectable object (person, bottle, chair, etc.)
- User-friendly interface with on-screen instructions

## Requirements
- Python 3
- OpenCV (`pip install opencv-python`)
- Docker (optional, for containerized deployment)
- YOLOv4-tiny weights and configuration files
- `classes.txt` file (already included)


## Download Required Files and Use

1. Clone the repository:
   ```bash
   git clone https://github.com/AnssiKuru/Dynamic-Object-Measurement.git

2. After cloning the repository, please download the following files manually and place them in the same directory as the Python script:

- [YOLOv4-tiny Weights](https://github.com/AnssiKuru/YOLO_Korkeusmittaus/raw/main/yolov4-tiny.weights)
- [YOLOv4-tiny Configuration](https://github.com/AnssiKuru/YOLO_Korkeusmittaus/raw/main/yolov4-tiny.cfg)


3. Run the script normally:
   ```bash
   python Dynamic_Object_Measurement.py
 

Follow the on-screen instructions and enjoy! 🎉

COMING SOON
(Optional) Instructions for creating and running a Docker container will be added shortly.

Example
Here is an example of live measurement in action:

![Live Measurement Example](live_measurement_example.png)

## Compatibility
Fully tested and optimized for Linux (Ubuntu, Jetson Nano).

Windows is supported but may require additional setup for OpenCV GUI (camera and window display).

Docker deployment is possible, but real-time GUI features (e.g., cv.imshow) require additional configuration (such as X11 forwarding).

## About
This project demonstrates dynamic real-time object measurement using a reference image and a YOLO object detector.
It is optimized for Jetson Nano and other small embedded systems, and designed for easy deployment on Linux devices.

Website: www.ml-art.fi

## Research Background
This project is a continuation and further development based on my Master's Thesis: Object Detection and Convolutional Neural Networks. (Currently in Finnish)

Thesis title: "Object Detection and Convolutional Neural Networks" (Currently in Finnish)

Link to thesis (in Finnish): [Object Detection and CNN Thesis](https://urn.fi/URN:NBN:fi:amk-2024060721931) 
