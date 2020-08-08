import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

trainDir = 'data/train'
testDir = 'data/test'
batchSize = 32
nEpochs = 1
imgHeight = 48
imgWidth = 48

nClasses = len(list(os.listdir(trainDir)))
numberOfTrainingImages = len(list(pathlib.Path(trainDir).glob('*/*.jpg')))
print(f'{nClasses = }')
print(f'{numberOfTrainingImages = }')

classWeights = {}
classNames = []
for sno, class_ in enumerate(os.listdir(trainDir)):
    bincount = len(list(os.listdir(os.path.join(trainDir, class_))))
    print(f'{class_ = }, {bincount = }')
    classWeights[sno] = numberOfTrainingImages / (nClasses * bincount)
    classNames.append(class_)

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

earlyStopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
stepsPerEpoch = trainGenerator.samples // trainGenerator.batch_size
validationSteps = valGenerator.samples // valGenerator.batch_size

history = model.fit(trainGenerator, epochs=nEpochs,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=valGenerator,
                    validation_steps=validationSteps,
                    callbacks=[earlyStopping],
                    class_weight=classWeights)

model.save('models/baseline.model')

testMetrics = model.evaluate(testGenerator)
print('\n\tTest Metrics:')
print(testMetrics)

y_pred = np.argmax(model.predict(testGenerator), axis=-1)
y_test = np.array([labels for _, labels in testGenerator])

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)
df = pd.DataFrame(matrix, index=classNames, columns=classNames)

fig = plt.figure(figsize=(12, 12))
hm = sns.heatmap(df, annot=True, cmap='coolwarm')
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
