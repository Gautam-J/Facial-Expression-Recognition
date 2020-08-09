import os
import time
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# TODO: Refactor! Add separate functions.

plt.style.use('seaborn')

baseDir = f'models/model_{int(time.time())}'
trainDir = 'data/train'
testDir = 'data/test'

if not os.path.exists(baseDir):
    os.makedirs(baseDir)

batchSize = 32
nEpochs = 5
imgHeight = 48
imgWidth = 48

nClasses = len(list(os.listdir(trainDir)))
numberOfTrainingImages = len(list(pathlib.Path(trainDir).glob('*/*.jpg')))
print(f'[INFO] {nClasses = }')
print(f'[INFO] {numberOfTrainingImages = }')

classWeights = {}
for sno, class_ in enumerate(os.listdir(trainDir)):
    bincount = len(list(os.listdir(os.path.join(trainDir, class_))))
    print(f'[INFO] {class_ = }, {bincount = }')
    classWeights[sno] = numberOfTrainingImages / (nClasses * bincount)

print(f'[INFO] {classWeights = }')

trainDatagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1 / 255.,
    validation_split=0.2)

testDatagen = ImageDataGenerator(rescale=1 / 255.)

print('[INFO] Train Generator')
trainGenerator = trainDatagen.flow_from_directory(
    directory=trainDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize,
    shuffle=True,
    subset='training',
    seed=42)

print('[INFO] Validation Generator')
valGenerator = trainDatagen.flow_from_directory(
    directory=trainDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize,
    shuffle=True,
    subset='validation',
    seed=42)

print('[INFO] Test Generator')
testGenerator = testDatagen.flow_from_directory(
    directory=testDir,
    target_size=(imgHeight, imgWidth),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batchSize)

model = Sequential()

model.add(Conv2D(64, (3, 1), padding='same', input_shape=(imgWidth, imgHeight, 1)))
model.add(Conv2D(64, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 1), padding='same'))
model.add(Conv2D(128, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 1), padding='same'))
model.add(Conv2D(256, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 1), padding='same'))
model.add(Conv2D(512, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nClasses))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'AUC'])

model.summary()
plot_model(model, to_file=f'{baseDir}/model_graph.png', show_shapes=True, dpi=200)

filePath = '{epoch:02d}_{val_loss:.4f}_{val_auc:.4f}.hdf5'
modelCheckpoint = ModelCheckpoint(f'{baseDir}/{filePath}', monitor='val_loss', save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=4)
reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=2)

stepsPerEpoch = trainGenerator.samples // trainGenerator.batch_size
validationSteps = valGenerator.samples // valGenerator.batch_size

history = model.fit(trainGenerator, epochs=nEpochs,
                    class_weight=classWeights,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=valGenerator,
                    validation_steps=validationSteps,
                    callbacks=[earlyStopping, modelCheckpoint, reduceLR])

testLoss, testAccuracy, testAUC = model.evaluate(testGenerator)
print('[INFO] Test Metrics:')
print(f'[INFO] {testLoss = }')
print(f'[INFO] {testAccuracy = }')
print(f'[INFO] {testAUC = }')

model.save(f'{baseDir}/final_model_{testLoss:.4f}_{testAccuracy:.4f}_{testAUC:.4f}.h5')

y_pred = np.argmax(model.predict(testGenerator), axis=-1)
y_test = testGenerator.classes
classLabels = list(testGenerator.class_indices.keys())

report = classification_report(y_test, y_pred, target_names=classLabels, output_dict=True)
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)

df = pd.DataFrame(report)
cr = sns.heatmap(df, annot=True, cmap='coolwarm')
cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.savefig(f'{baseDir}/classification_report.png')
plt.show()

df = pd.DataFrame(matrix, index=classLabels, columns=classLabels)
hm = sns.heatmap(df, annot=True, cmap='coolwarm')
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.savefig(f'{baseDir}/confusion_matrix.png')
plt.show()

plt.subplot(131)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.subplot(132)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(133)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Area Under ROC')
plt.legend()

plt.savefig(f'{baseDir}/training_history.png')
plt.show()
