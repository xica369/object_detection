#!/usr/bin/env python3

import cv2
from scripts.yolo import Yolo
# from yolo import Yolo
import numpy as np
import os


def prediction(path_app='../'):    
    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    
    h5_path = os.path.join(path_app, 'data/yolo.h5')
    classes_path = os.path.join(path_app, 'data/coco_classes.txt')
    _yolo = Yolo(h5_path, classes_path, 0.6, 0.5, anchors)
    predictions, image_paths = _yolo.predict(path_app=path_app)

if __name__ == '__main__':
    prediction()