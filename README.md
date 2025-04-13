# Dynamic Object Measurement

This project automatically detects objects from a reference image and allows real-time distance and height measurement using YOLOv4-tiny and OpenCV.

## Features
- Automatic reference image capture
- Dynamic object detection (best match selected automatically)
- Real-time object measurement
- Works with any detectable object (person, bottle, chair, etc.)
- User-friendly interface with on-screen instructions
- Website watermark displayed: [www.ml-art.fi](https://www.ml-art.fi)

## Requirements
- Python 3
- OpenCV (`pip install opencv-python`)
- Docker (optional, for containerized deployment)
- YOLOv4-tiny weights and configuration files
- `classes.txt` file (already included)

## Download Required Files

After cloning the repository, please download the following files manually and place them in the same directory as the Python script:

- [YOLOv4-tiny Weights](https://github.com/AnssiKuru/YOLO_Korkeusmittaus/raw/main/yolov4-tiny.weights)
- [YOLOv4-tiny Configuration](https://github.com/AnssiKuru/YOLO_Korkeusmittaus/raw/main/yolov4-tiny.cfg)

*Note: The `classes.txt` file is already included in this repository.*

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AnssiKuru/Dynamic-Object-Measurement.git
Download the YOLOv4-tiny files (weights and cfg) from the links above.

Place the files in the same folder as the Python script.

(Optional) Create and run a Docker container (instructions coming soon).

Run the script normally:

    ```bash
   python Dynamic_Object_Measurement.py

Follow the on-screen instructions:


Example
Here is an example of live measurement in action:

![Live Measurement Example](live_measurement_example.png)

About
This project demonstrates dynamic real-time object measurement using a reference image and a YOLO object detector. It is optimized for Jetson Nano and other small embedded systems.

Website: www.ml-art.fi
