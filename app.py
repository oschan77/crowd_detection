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


class Person:
    def __init__(self, bbox: List[float], depth: float = None):
        self.x1, self.y1, self.x2, self.y2 = map(int, bbox)
        self.center = (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
        self.depth = depth

    def draw(self, image: np.ndarray) -> np.ndarray:
        return cv2.rectangle(
            image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2
        )


class Crowd:
    def __init__(self, resolution: Tuple[int]):
        self.people = []
        self.x1, self.y1, self.x2, self.y2 = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )

        self.resolution = resolution

    def is_close(
        self, person: Person, threshold_x: float = 0.04, threshold_z: float = 0.08
    ) -> bool:
        for other in self.people:
            z_diff = abs(person.depth - other.depth) / 255.0

            if person.x2 < other.x1:
                x_diff = abs(other.x1 - person.x2) / self.resolution[0]
            elif other.x2 < person.x1:
                x_diff = abs(person.x1 - other.x2) / self.resolution[0]
            else:
                x_diff = 0

            # print(f"x_diff: {x_diff}, z_diff: {z_diff}")

            if x_diff < threshold_x and z_diff < threshold_z:
                return True

    def add_person(self, person: Person):
        self.people.append(person)
        self.update_bbox([person.x1, person.y1, person.x2, person.y2])

    def merge_crowd(self, crowd: "Crowd"):
        self.people.extend(crowd.people)
        self.update_bbox([crowd.x1, crowd.y1, crowd.x2, crowd.y2])

    def update_bbox(self, bbox: List[int]):
        if bbox[0] < self.x1:
            self.x1 = int(bbox[0])
        if bbox[1] < self.y1:
            self.y1 = int(bbox[1])
        if bbox[2] > self.x2:
            self.x2 = int(bbox[2])
        if bbox[3] > self.y2:
            self.y2 = int(bbox[3])

    def draw(self, image: np.ndarray, nPeople: int) -> np.ndarray:
        img = cv2.rectangle(
            image, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2
        )

        img = cv2.putText(
            img,
            str(nPeople),
            (self.x1, self.y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )

        return img


def init_model():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    resolution = (1920 // 2, 1080 // 2)
    rs_config.enable_stream(
        rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 15
    )
    rs_pipeline.start(rs_config)

    depth_model = transformers.pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-base-hf",
        device=0 if torch.cuda.is_available() else -1,
    )
    # od_model = YOLO("yolov8x.pt").to("cuda" if torch.cuda.is_available() else "cpu")
    od_model = YOLO("yolov8x.engine")

    while True:
        rs_frames = rs_pipeline.wait_for_frames()
        rs_color_frame = rs_frames.get_color_frame()
        if not rs_color_frame:
            continue

        color_img = np.asanyarray(rs_color_frame.get_data())

        # OD
        od_preds = od_model.predict(color_img, classes=[0], conf=0.6, device="cuda")
        nPeople = len(od_preds[0].boxes)
        od_img = cv2.putText(
            color_img,
            f"Total Number of people: {nPeople}",
            (int(resolution[0] // 2 - 200), 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # DE
        depth_img = np.array(
            depth_model(Image.fromarray(color_img))["depth"].convert("L")
        )

        # Find Crowds
        crowds = []
        bboxes = od_preds[0].boxes.xyxy
        for bbox in bboxes:
            person = Person(bbox)
            depth = depth_img[person.y1 : person.y2, person.x1 : person.x2]
            depth = depth[depth != 0]
            person.depth = np.mean(depth)

            # print(f"Depth: {person.depth}")

            od_img = person.draw(od_img)

            if len(crowds) == 0:
                crowd = Crowd(resolution)
                crowd.add_person(person)
                crowds.append(crowd)
            else:
                closedCrowdIdx = [
                    crowdIdx
                    for crowdIdx in range(len(crowds))
                    if crowd.is_close(person)
                ]
                if len(closedCrowdIdx) == 0:
                    crowd = Crowd(resolution)
                    crowd.add_person(person)
                    crowds.append(crowd)
                elif len(closedCrowdIdx) == 1:
                    crowd = crowds[closedCrowdIdx[0]]
                    crowd.add_person(person)
                else:
                    main_crowd = crowds[closedCrowdIdx[0]]
                    main_crowd.add_person(person)

                    for crowdIdx in closedCrowdIdx[-1:0:-1]:
                        main_crowd.merge_crowd(crowds[crowdIdx])
                        crowds.pop(crowdIdx)

        for crowd in crowds:
            if len(crowd.people) > 1:
                od_img = crowd.draw(od_img, len(crowd.people))

        od_img = cv2.resize(od_img, (resolution[0], resolution[1]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, buffer = cv2.imencode(".jpg", od_img)
        od_img = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + od_img + b"\r\n")

    # Clear resource.
    cv2.destroyAllWindows()


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
