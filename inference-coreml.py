import coremltools
from PIL import Image
import numpy as np
import core.min_utils as utils
from core.config import cfg
import cv2

input_size = 416
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

model = coremltools.models.MLModel('models/yolov4_8.mlmodel')

## Video frame for comparing with multi-stage ios env
cap = cv2.VideoCapture("/Users/mac/Downloads/food_test3-2020-07-12_17.50.09.mp4")
# cap.set(1, cap.get(7)-3)
ret, im_rgb = cap.read()
im_rgb = cv2.cvtColor(cv2.resize(im_rgb.copy(), (416,416)), cv2.COLOR_BGR2RGB)
original_image = Image.fromarray(im_rgb)

## Image for testing
# original_image = Image.open('../group_21.jpg').resize((416,416))
original_image_size = original_image.size[:2]
img = original_image.copy()

## Pre-processing is added to the coreml model
#img = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
preds = model.predict({'input_1':img})
_ = [print(preds[key].shape,key) for key in preds.keys()]
preds = [ preds["Identity"], preds["Identity_1"], preds["Identity_2"] ]

pred_bbox = utils.postprocess_bbbox(preds, ANCHORS, STRIDES, XYSCALE)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.213, method='nms')

for i, bbox in enumerate(bboxes):
    coor = np.array(bbox[:4], dtype=np.int32)
    fontScale = 0.5
    score = bbox[4]
    class_ind = int(bbox[5])
    c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
    print( class_ind , " - ", score, " - " ,(c1[0]+c2[0])/2 , (c1[1]+c2[1])/2 , c2[0]-c1[0], c2[1]-c1[1])

original_image = np.array(original_image.copy())
image = utils.draw_bbox(original_image, bboxes)
# Image.show(image)
cv2.imwrite("temp_det.jpg",image)
