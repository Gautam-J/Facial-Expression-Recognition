import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

plt.style.use('seaborn')

trainDir = 'data/train'
testDir = 'data/test'
batchSize = 32
nEpochs = 5
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
model.add(Conv2D(32, 3, input_shape=(imgWidth, imgHeight, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, 2, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'AUC'])

model.summary()
plot_model(model, to_file='models/baseline_model.png', show_shapes=True, dpi=200)

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
print('Test Loss =', testMetrics[0])
print('Test Accuracy =', testMetrics[1])
print('Test AUC =', testMetrics[2])

y_pred = np.argmax(model.predict(testGenerator), axis=-1)
y_test = testGenerator.classes
classLabels = list(testGenerator.class_indices.keys())

report = classification_report(y_test, y_pred, target_names=classLabels, output_dict=True)
print('\n\tClassification Report')
print(report)

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)
print('\n\tConfusion Matrix')
print(matrix)

df = pd.DataFrame(report)
cr = sns.heatmap(df, annot=True, cmap='coolwarm')
cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.savefig('models/baseline_cr.png')
plt.show()

df = pd.DataFrame(matrix, index=classLabels, columns=classLabels)
hm = sns.heatmap(df, annot=True, cmap='coolwarm')
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.savefig('models/baseline_cm.png')
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

plt.savefig('models/baseline_history.png')
plt.show()
