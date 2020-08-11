import numpy as np

from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report

from model_configs import buildModel, getModelCallbacks
from utils import getClassWeights, createBaseDir, getNumberOfClasses
from generators import getTrainGenerator, getValGenerator, getTestGenerator

from visualizations import (plotClassificationReport,
                            plotConfusionMatrix,
                            plotTrainingHistory)

baseDir = createBaseDir()
trainDir = 'data/train'
testDir = 'data/test'

batchSize = 32
nEpochs = 100
imgHeight = 48
imgWidth = 48

classWeights = getClassWeights(trainDir)

trainGenerator = getTrainGenerator(trainDir,
                                   (imgWidth, imgHeight),
                                   batchSize)

valGenerator = getValGenerator(trainDir,
                               (imgWidth, imgHeight),
                               batchSize)

testGenerator = getTestGenerator(testDir,
                                 (imgWidth, imgHeight),
                                 batchSize)

nClasses = getNumberOfClasses(trainDir)
model = buildModel(input_shape=(imgWidth, imgHeight, 1),
                   number_of_classes=nClasses)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'AUC'])

plot_model(model, to_file=f'{baseDir}/model_graph.png', show_shapes=True, dpi=200)

stepsPerEpoch = trainGenerator.samples // trainGenerator.batch_size
validationSteps = valGenerator.samples // valGenerator.batch_size
callBacks = getModelCallbacks(baseDir)

history = model.fit(trainGenerator, epochs=nEpochs,
                    class_weight=classWeights,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=valGenerator,
                    validation_steps=validationSteps,
                    callbacks=callBacks)

testLoss, testAccuracy, testAUC = model.evaluate(testGenerator)
print('[INFO] Test Metrics:')
print(f'[INFO] {testLoss = }')
print(f'[INFO] {testAccuracy = }')
print(f'[INFO] {testAUC = }')

model.save(f'{baseDir}/final_model_{testLoss:.4f}_{testAccuracy:.4f}_{testAUC:.4f}.h5')

y_pred = np.argmax(model.predict(testGenerator), axis=-1)  # model.predict_classes()
y_test = testGenerator.classes
classLabels = list(testGenerator.class_indices.keys())

report = classification_report(y_test, y_pred,
                               target_names=classLabels,
                               output_dict=True)

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)  # normalize confusion matrix

print('[INFO] Plotting classification report')
plotClassificationReport(report, baseDir)

print('[INFO] Plotting confusion matrix')
plotConfusionMatrix(matrix, classLabels, baseDir)

print('[INFO] Plotting training history')
plotTrainingHistory(history, baseDir)
