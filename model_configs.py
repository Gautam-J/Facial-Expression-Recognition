from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


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


def getModelCallbacks(directory):
    filePath = '{epoch:02d}_{val_loss:.4f}_{val_auc:.4f}.hdf5'

    modelCheckpoint = ModelCheckpoint(f'{directory}/{filePath}',
                                      monitor='val_loss',
                                      save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=4)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=2)

    return [modelCheckpoint,
            earlyStopping,
            reduceLR]
