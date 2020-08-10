import os
import time
import pathlib


def createBaseDir():
    global baseDir
    baseDir = f'models/model_{int(time.time())}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)


def getBaseDir():
    return baseDir


def getNumberOfClasses(path_to_train_dir):
    return len(list(os.listdir(path_to_train_dir)))


def getNumberOfTrainingImages(path_to_train_dir):
    return len(list(pathlib.Path(path_to_train_dir).glob('*/*.jpg')))


def getClassWeights(path_to_train_dir):
    global nClasses

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
