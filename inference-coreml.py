import coremltools
from PIL import Image
import numpy as np
import core.min_utils as utils
from core.config import cfg

input_size = 416
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

model = coremltools.models.MLModel('test1.mlmodel')

original_image = np.array(Image.open('data/group_32.jpg'))
original_image_size = original_image.shape[:2]

img = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
preds = model.predict({'input_1':np.array([img.astype(np.float32)])})
print(preds.values())

pred_bbox = utils.postprocess_bbbox(preds, ANCHORS, STRIDES, XYSCALE)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.213, method='nms')

image = utils.draw_bbox(original_image, bboxes)
cv2.imwrite("temp_det.jpg")