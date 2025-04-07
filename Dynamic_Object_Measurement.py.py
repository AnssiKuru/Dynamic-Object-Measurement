# -*- coding: utf-8 -*-
"""
Full Dynamic Object Reference Measurement and Live Detection System
Author: Anssi Kuru
Website: https://www.ml-art.fi/
"""

import cv2 as cv
import time

# Font for overlay text
FONT = cv.FONT_HERSHEY_SIMPLEX

# Object detection settings
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
COLORS = [(255,0,0), (255,0,255), (0,255,255), (255,255,0), (0,255,0), (255,0,0)]

# Load class names
with open("classes.txt", "r") as f:
    CLASS_NAMES = [cname.strip() for cname in f.readlines()]

def capture_reference_image_automatically(camera, filename="reference.jpg", wait_time=10):
    print("\nReference image will be captured automatically in 10 seconds...")
    start_time = time.time()
    clean_frame = None

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error reading from camera.")
            break

        elapsed_time = int(time.time() - start_time)
        remaining_time = wait_time - elapsed_time

        clean_frame = frame.copy()

        if remaining_time > 0:
            cv.putText(frame, "Move to the correct distance (e.g., 150 cm)", (10, 30), FONT, 0.7, (0, 255, 0), 2)
            cv.putText(frame, "Make sure the whole object fits inside the frame", (10, 60), FONT, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f"Capturing image in {remaining_time} seconds...", (10, 90), FONT, 0.8, (0, 0, 255), 2)

        cv.imshow('Reference Capture', frame)

        if remaining_time <= 0:
            cv.imwrite(filename, clean_frame)
            print(f"\nReference image saved as: {filename}")
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("\nCancelled capturing reference image.")
            break

    cv.destroyAllWindows()

def load_yolo_model():
    net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    return model

def detect_objects(model, image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return classes, scores, boxes

def select_best_object(classes, scores, boxes):
    if len(scores) == 0:
        return None
    max_index = scores.argmax()
    return classes[max_index], scores[max_index], boxes[max_index]

def ask_reference_dimensions(image, box, label):
    user_inputs = ["", "", ""]
    fields = [
        f"Detected object: {label}",
        "Enter distance from camera (cm): ",
        "Enter real object height (cm): ",
        "Enter real object width (cm): "
    ]
    current_field = 1  # Skip showing label, start with distance input

    while True:
        frame = image.copy()
        x, y, w, h = box
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f"{label}", (x, y-10), FONT, 0.7, (0, 255, 0), 2)

        cv.putText(frame, fields[current_field], (10, 30), FONT, 0.7, (0, 255, 255), 2)
        cv.putText(frame, user_inputs[current_field-1], (10, 70), FONT, 1.0, (255, 0, 0), 2)

        cv.imshow("Reference Setup", frame)

        key = cv.waitKey(1) & 0xFF

        if key >= ord('0') and key <= ord('9'):
            user_inputs[current_field-1] += chr(key)
        elif key == ord('.') or key == ord(','):
            user_inputs[current_field-1] += '.'
        elif key == 8:
            user_inputs[current_field-1] = user_inputs[current_field-1][:-1]
        elif key == 13:
            current_field += 1
            if current_field >= len(fields):
                break
        elif key == ord('q'):
            print("Cancelled.")
            cv.destroyAllWindows()
            return None, None, None, None

    cv.destroyAllWindows()

    distance = float(user_inputs[0])
    real_height = float(user_inputs[1])
    real_width = float(user_inputs[2])

    return distance, real_height, real_width, box

def measure_in_real_time(camera, model, distance, real_height, real_width, reference_width_pixels, label):
    print("\nStarting real-time measurement... Press 'q' to quit.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error reading frame.")
            break

        classes, scores, boxes = detect_objects(model, frame)

        for (classid, score, box) in zip(classes, scores, boxes):
            x, y, w, h = box
            class_name = CLASS_NAMES[classid]

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, class_name, (x, y-10), FONT, 0.7, (0, 255, 0), 2)

            if class_name == label:
                focal_length = (reference_width_pixels * distance) / real_width
                measured_distance = (real_width * focal_length) / w
                measured_height = (h / reference_width_pixels) * real_height

                cv.putText(frame, f"Height: {round(measured_height,2)} cm", (x, y-50), FONT, 0.7, (0, 255, 255), 2)
                cv.putText(frame, f"Distance: {round(measured_distance,2)} cm", (x, y-30), FONT, 0.7, (0, 255, 255), 2)

        # Bottom website watermark
        cv.putText(frame, "www.ml-art.fi", (10, frame.shape[0] - 10), FONT, 0.7, (255, 255, 255), 2)

        # Top quit instruction
        cv.putText(frame, "Press 'q' to quit measurement", (10, 30), FONT, 0.7, (0, 0, 255), 2)

        cv.imshow("Live Measurement", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("\nMeasurement stopped by user.")
            break

    cv.destroyAllWindows()

def main():
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    capture_reference_image_automatically(camera)
    model = load_yolo_model()

    ref_img = cv.imread("reference.jpg")
    classes, scores, boxes = detect_objects(model, ref_img)
    selected = select_best_object(classes, scores, boxes)

    if selected is None:
        print("No object detected!")
        camera.release()
        return

    classid, score, box = selected
    label = CLASS_NAMES[classid]

    distance, real_height, real_width, box = ask_reference_dimensions(ref_img, box, label)
    if distance is None:
        camera.release()
        return

    reference_width_pixels = box[2]  # Width in pixels

    measure_in_real_time(camera, model, distance, real_height, real_width, reference_width_pixels, label)

    camera.release()

if __name__ == "__main__":
    main()
