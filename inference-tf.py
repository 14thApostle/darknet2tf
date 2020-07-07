import time
import core.min_utils as utils
from PIL import Image
from core.config import cfg
import core.common

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

input_size = 416
image_path = "/home/lordgrim/Work/VisioLab/smaller-data/Images/group_2028.jpg"

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

model_path = "models/yolov4_saved"
model = load_model(model_path)
model.summary()

pred_bbox = model.predict(image_data)

pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.213, method='nms')

image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()
