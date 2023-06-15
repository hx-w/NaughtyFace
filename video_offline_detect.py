# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import random

from face_age_estimator import estimate_age

def _img2video(imgs: list, video_path: str, fps: int, shape: tuple):
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, shape)
 
    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()


def _run_video(video_capture: cv2.VideoCapture, video_path: str, sample_rate: float):
    video_capture.open(video_path)
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    _shape = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps: {}, frame_count: {}".format(_fps, _frame_count))

    frames = []    
    for _ in range(int(_frame_count)):
        ret, frame = video_capture.read()
        if not ret: break
        frames.append(frame)

    frames = np.array(frames)
    randIndex = random.sample(range(int(_frame_count)), int(_frame_count * sample_rate))
    randIndex.sort()

    frames = frames[randIndex]

    edited_imgs, ages, r_or_f = estimate_age(frames)

    avg_age = np.mean(ages, axis=0)
    print("avg_age: {}".format(avg_age), 'Fake' if np.any(False in r_or_f) else 'Real')

    _img2video(edited_imgs, video_path.replace('.mp4', '_pred.mp4'), _fps, _shape)


def detect_main(video_path: str, sample_rate: float):
    if not os.path.exists(video_path):
        raise ValueError("video path not exists: {}".format(video_path))
    
    if sample_rate <= 0.0 or sample_rate > 1.0:
        raise ValueError("sample rate should be in (0.0, 1.0]")

    video_capture = cv2.VideoCapture()
    _run_video(video_capture, video_path, sample_rate)


def get_args():
    desc = 'A Naughty Face Detector Demo'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--inputs",
        "-i",
        type=str,
        required=True,
        help="video path"
    )
    parser.add_argument(
        "--sample_rate",
        "-s",
        type=float,
        default=0.5,
        help="sample rate"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    detect_main(args.inputs, args.sample_rate)