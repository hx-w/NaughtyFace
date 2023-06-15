# -*- coding: utf-8 -*-

import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from age_estimator.model import get_model
from age_estimator.defaults import _C as cfg

from face_anti_spoofing import check_spoofing

model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


resume_path = Path(__file__).resolve().parent.joinpath("age_estimator", "misc", "epoch044_0.02343_3.9984.pth")

if not resume_path.is_file():
    print(f"=> model path is not set; start downloading trained model to {resume_path}")
    url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
    urllib.request.urlretrieve(url, str(resume_path))
    print("=> download finished")

if Path(resume_path).is_file():
    print("=> loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(resume_path))
else:
    raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

if device == "cuda":
    cudnn.benchmark = True

model.eval()
margin = 4
detector = dlib.get_frontal_face_detector()
img_size = cfg.MODEL.IMG_SIZE

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1, clr=(255, 255, 255)):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, clr, thickness, lineType=cv2.LINE_AA)


def estimate_age(imgs) -> int:
    edited_imgs, pred_ages, pred_real_or_fake = [], [], []
    with torch.no_grad():
        for img in tqdm(imgs, desc="Estimating age"):
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # anti-spoofing
                real_or_fake = list(map(check_spoofing, faces))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)

                # draw results
                for i, d in enumerate(detected):
                    label = "{} {}".format(int(predicted_ages[i]), 'REAL' if real_or_fake[i] else 'FAKE')
                    draw_label(img, (d.left(), d.top()), label, font_scale=1.0, clr=(255, 0, 0) if real_or_fake[i] else (0, 255, 0))

                edited_imgs.append(img)
                pred_ages.append(predicted_ages)
                pred_real_or_fake.append(real_or_fake)

    return edited_imgs, np.array(pred_ages), np.array(pred_real_or_fake)
