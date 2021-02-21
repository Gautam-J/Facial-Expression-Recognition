import os
import time
import pathlib


def createBaseDir():
    """Creates a directory for storing all the models.

    Returns:
        str: Path to the directory created.
    """

    baseDir = f'models/model_{int(time.time())}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    return baseDir


def getNumberOfClasses(path_to_train_dir):
    return len(list(os.listdir(path_to_train_dir)))


def getNumberOfTrainingImages(path_to_train_dir):
    return len(list(pathlib.Path(path_to_train_dir).glob('*/*.jpg')))


def getClassWeights(path_to_train_dir):
    """Get class weights for the training dataset.

    Args:
        path_to_train_dir (str): Path where the train images are stored.

    Returns:
        dict: Contains class weights for each class.
    """

    nClasses = getNumberOfClasses(path_to_train_dir)
    numberOfTrainingImages = getNumberOfTrainingImages(path_to_train_dir)
    classList = sorted(os.listdir(path_to_train_dir))

    classWeights = {}
    for sno, className in enumerate(classList):
        bincount = len(list(os.listdir(os.path.join(path_to_train_dir, className))))
        classWeights[sno] = numberOfTrainingImages / (nClasses * bincount)

    return classWeights
