import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

import core.min_utils as utils
from core.config import cfg

model_path = "/home/lordgrim/Work/VisioLab/tensorflow-yolov4-tflite/data/1-yolov4.tflite"
input_size = 416
image_path = "/home/lordgrim/Work/VisioLab/smaller-data/Images/group_2028.jpg"

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]


image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
a = time.time()
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.213, method='nms')
# bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
print(bboxes)

image = utils.draw_bbox(original_image, bboxes)
image = cv2.imshow("out",image)
cv2.waitKey(0)
