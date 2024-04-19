import argparse
import ctypes as ct
import datetime
import enum
import os
import sys
import time
from ctypes.util import find_library
from turtle import color
from typing import List, Tuple
from unittest import result

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch
import transformers
from flask import Flask, Response, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

app = Flask(__name__)

alarm = False
thermal_map = None


def plot_temperature(im, box, temperature, txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    sf = lw / 3
    tf = max(lw - 1, 1)
    """Add one xyxy box to image with label."""
    label = "Temperature: {}".format(str(int(temperature)))
    if not isinstance(box, list):
        box = box.tolist()

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
        0
    ]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        im,
        label,
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        sf,
        txt_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return im


def plot_util(image, h, w, box, temperature, except_case=False):
    fs = int((h + w) * 0.01)  # font size
    annotator = Annotator(image, line_width=round(fs / 6), font_size=fs * 10, pil=False)
    if temperature > 100 and not except_case:
        color = (255, 10, 10)
        global alarm
        alarm = True
        send_signal()
    else:
        color = (10, 255, 10)
    annotator.box_label(
        box, "Temperature: " + str(int(temperature)), color=color, temperature=True
    )
    return annotator.im


def send_signal():
    response = requests.get(
        "http://127.0.0.1:5000/alarm"
    )  # Replace with your Flask app's URL
    if response.status_code == 200:
        print("Signal sent successfully")
    else:
        print("Failed to send signal")


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        )
    )


def init_model():
    # od_model = YOLO("yolov8n.pt").to("cuda")
    od_model = YOLO("yolov8n.pt")
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    # resolution = (1280, 720)
    resolution = (424, 240)
    rs_config.enable_stream(
        rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 5
    )
    rs_pipeline.start(rs_config)

    try:
        while True:
            rs_frames = rs_pipeline.wait_for_frames()
            rs_color_frame = rs_frames.get_color_frame()
            if not rs_color_frame:
                continue

            color_img = np.asanyarray(rs_color_frame.get_data())
            # od_preds = od_model.predict(color_img, classes=[0], conf=0.4, device="cuda")
            od_preds = od_model.predict(color_img, classes=[0], conf=0.4, device="cpu")
            nPeople = len(od_preds[0].boxes)

            color_img = od_preds[0].plot()
            color_img = cv2.putText(
                color_img,
                f"Number of people: {nPeople}",
                (int(resolution[0] // 2 - 100), 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            if nPeople >= 3:
                color_img = cv2.putText(
                    color_img,
                    f"Crowd Detected!",
                    (int(resolution[0] // 2 - 100), 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", color_img)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        rs_pipeline.stop()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(init_model(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alarm")
def update_image():
    # Your Python code logic here
    # Check the condition and send the signal when reached
    global alarm
    if alarm:
        # Perform any necessary operations
        # ...
        return jsonify(success=True)
    else:
        return jsonify(success=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
