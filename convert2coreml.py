import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from core.yolov4 import YOLOv4
from core.config import cfg

from coremltools.models import MLModel
import matplotlib.pyplot as plt


model_path = "models/final_tf_model5000"
model = load_model(model_path)
model.summary()

import coremltools as ct

# convert to Core ML
mlmodel = ct.convert(model)
mlmodel.save("models/test1.mlmodel")