import os
import time
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# TODO: Group functions and move them to separate files

plt.style.use('seaborn')


def createBaseDir():
    global baseDir
    baseDir = f'models/model_{int(time.time())}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    return baseDir


def getNumberOfClasses(path_to_train_dir):
    return len(list(os.listdir(path_to_train_dir)))


def getNumberOfTrainingImages(path_to_train_dir):
    return len(list(pathlib.Path(path_to_train_dir).glob('*/*.jpg')))


def getClassWeights(path_to_train_dir):

    nClasses = getNumberOfClasses(path_to_train_dir)
    numberOfTrainingImages = getNumberOfTrainingImages(path_to_train_dir)
    print(f'[INFO] {nClasses = }')
    print(f'[INFO] {numberOfTrainingImages = }')

    classWeights = {}
    for sno, className in enumerate(os.listdir(path_to_train_dir)):
        bincount = len(list(os.listdir(os.path.join(path_to_train_dir, className))))
        classWeights[sno] = numberOfTrainingImages / (nClasses * bincount)
        print(f'[INFO] {className = }, {bincount = }')

    return classWeights


def getTrainDatagen():
    return ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.2,
        rescale=1 / 255.,
        validation_split=0.2)


def getTestDatagen():
    return ImageDataGenerator(rescale=1 / 255.)


def getTrainGenerator(directory, target_size, batch_size):
    global trainDatagen
    trainDatagen = getTrainDatagen()

    print('[INFO] Train Generator')
    trainGenerator = trainDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training',
        seed=42)

    return trainGenerator


def getValGenerator(directory, target_size, batch_size):

    print('[INFO] Validation Generator')
    valGenerator = trainDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='validation',
        seed=42)

    return valGenerator


def getTestGenerator(directory, target_size, batch_size):
    global testDatagen
    testDatagen = getTestDatagen()

    print('[INFO] Test Generator')
    testGenerator = testDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size)

    return testGenerator


def buildModel(input_shape, number_of_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 1), padding='same', input_shape=input_shape))
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

    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))

    model.summary()

    return model


def getModelCallbacks():
    filePath = '{epoch:02d}_{val_loss:.4f}_{val_auc:.4f}.hdf5'

    modelCheckpoint = ModelCheckpoint(f'{baseDir}/{filePath}',
                                      monitor='val_loss',
                                      save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=4)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=2)

    return [modelCheckpoint,
            earlyStopping,
            reduceLR]


def plotClassificationReport(report):
    df = pd.DataFrame(report).T

    cr = sns.heatmap(df, annot=True, cmap='coolwarm')
    cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.savefig(f'{baseDir}/classification_report.png')


def plotConfusionMatrix(matrix, class_labels):
    df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

    hm = sns.heatmap(df, annot=True, cmap='coolwarm')
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.savefig(f'{baseDir}/confusion_matrix.png')


def plotTrainingHistoryLoss(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f'{baseDir}/training_history_loss.png')


def plotTrainingHistoryAccuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f'{baseDir}/training_history_accuracy.png')


def plotTrainingHistoryAUC(history):
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Area Under ROC')
    plt.legend()
    plt.savefig(f'{baseDir}/training_history_AUC.png')


def plotTrainingHistory(history):
    print('[INFO] Plotting training history')

    plotTrainingHistoryAccuracy(history)
    plotTrainingHistoryLoss(history)
    plotTrainingHistoryAUC(history)
