# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import warnings
import time

from anti_spoofing.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.generate_patches import CropImage
from anti_spoofing.utility import parse_model_name
warnings.filterwarnings('ignore')


model = AntiSpoofPredict(0)
image_cropper = CropImage()

model_dir = './resources/anti_spoof_models'

def check_spoofing(img) -> list:
    prediction = np.zeros((1, 3))
    test_speed = 0
    image_bbox = np.array([0, 0, img.shape[1], img.shape[0]])
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2

    return label == 1