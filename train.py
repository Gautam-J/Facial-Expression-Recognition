import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

trainDir = 'data/train'
testDir = 'data/test'
BATCH_SIZE = 32
IMG_HEIGHT = 48
IMG_WIDTH = 48

n_classes = len(list(os.listdir(trainDir)))
numberOfTrainingImages = len(list(pathlib.Path(trainDir).glob('*/*.jpg')))
print(f'{n_classes = }')
print(f'{numberOfTrainingImages = }')

class_weights = {}
for sno, class_ in enumerate(os.listdir(trainDir)):
    bincount = len(list(os.listdir(os.path.join(trainDir, class_))))
    print(f'{class_ = }, {bincount = }')
    class_weights[sno] = numberOfTrainingImages / (n_classes * bincount)

print(f'{class_weights = }')
