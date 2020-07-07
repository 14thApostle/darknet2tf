import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

from core.yolov4 import YOLOv4
from core.config import cfg

from coremltools.models import MLModel
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output',required=True)

args = parser.parse_args()

model_path = args.input
model = load_model(model_path)
model.summary()

import coremltools as ct

# convert to Core ML
mlmodel = ct.convert(model)
mlmodel.save(args.output)