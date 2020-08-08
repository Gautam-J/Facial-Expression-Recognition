import os
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten

trainDir = 'data/train'
testDir = 'data/test'
batchSize = 32
nEpochs = 2
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
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rescale=1 / 255.,
    validation_split=0.2)

testDatagen = ImageDataGenerator(rescale=1 / 255.)

print('Train Generator')
trainGenerator = trainDatagen.flow_from_directory(
    directory=trainDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize,
    shuffle=True,
    subset='training',
    seed=42)

print('Validation Generator')
valGenerator = trainDatagen.flow_from_directory(
    directory=trainDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize,
    shuffle=True,
    subset='validation',
    seed=42)

print('Test Generator')
testGenerator = testDatagen.flow_from_directory(
    directory=testDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize)

model = Sequential()
model.add(Conv2D(64, 2, input_shape=(imgWidth, imgHeight, 1), activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'AUC'])

model.summary()

stepsPerEpoch = trainGenerator.samples // trainGenerator.batch_size
validationSteps = valGenerator.samples // valGenerator.batch_size

history = model.fit(trainGenerator, epochs=nEpochs,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=valGenerator,
                    validation_steps=validationSteps,
                    class_weight=classWeights)

model.save('models/baseline.model')

testMetrics = model.evaluate(testGenerator)
print('\n\tTest Metrics:')
print(testMetrics)
