from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
