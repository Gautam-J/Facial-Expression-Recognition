import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

trainDir = 'data/train'
testDir = 'data/test'
batchSize = 32
imgHeight = 48
imgWidth = 48

nClasses = len(list(os.listdir(trainDir)))
numberOfTrainingImages = len(list(pathlib.Path(trainDir).glob('*/*.jpg')))
print(f'{nClasses = }')
print(f'{numberOfTrainingImages = }')

classWeights = {}
for sno, class_ in enumerate(os.listdir(trainDir)):
    bincount = len(list(os.listdir(os.path.join(trainDir, class_))))
    print(f'{class_ = }, {bincount = }')
    classWeights[sno] = numberOfTrainingImages / (nClasses * bincount)

print(f'{classWeights = }')

trainDatagen = ImageDataGenerator(
    zca_whitening=True,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rescale=1 / 255.,
    validation_split=0.2)

testDatagen = ImageDataGenerator(rescale=1 / 255.)

trainGenerator = trainDatagen.flow_from_directory(
    directory=trainDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=batchSize,
    shuffle=True,
    seed=42)

testGenerator = testDatagen.flow_from_directory(
    directory=testDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=batchSize)
