import keras

from metrics import *
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.advanced_activations import ReLU, ELU
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D


class HJModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=None, channel_order='channels_first'):

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', data_format=channel_order, name='conv1'))

        model.add(Conv2D(128, (3, 3), activation='relu', name='conv2', data_format=channel_order))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

        model.add(Conv2D(256, (3, 3), activation='relu', name='conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

        model.add(Conv2D(512, (3, 3), activation='relu', name='conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

        model.add(Flatten())

        model.add(Dense(1024, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(512, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(256, kernel_initializer='uniform', activation='relu'))

        model.add(Dense(number_of_classes, kernel_initializer='uniform', activation='softmax'))

        sgd = SGD(lr=0.08, decay=1e-10, momentum=0.9, nesterov=False)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model


class BHModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=114, channel_order='channels_first'):

        model = Sequential()
        model.add(Conv2D(32, (1, 5), padding='same', kernel_initializer='he_normal', input_shape=input_shape, name='conv1'))
        model.add(ELU())
        model.add(BatchNormalization())
        # model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2'))
        # model.add(ELU())
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        # model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
        # model.add(ELU())
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(number_of_classes, kernel_initializer='he_normal'))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model

class NewBHModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=114, channel_order='channels_first'):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='elu', data_format=channel_order))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Conv2D(128, (3, 3), activation='elu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(512, kernel_initializer='uniform', activation='elu'))
        model.add(Dense(number_of_classes, kernel_initializer='uniform', activation='softmax'))

        sgd = SGD(lr=0.03, decay=0.01 / 40, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model


class JHModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=2, channel_order='channels_first'):

        model = Sequential()

        model.add(Conv2D(80, (5, 5), input_shape=input_shape, data_format=channel_order, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(96, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(number_of_classes, activation='softmax'))

        sgd = SGD(lr=0.08, decay=0.01 / 40, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model


class OneJHModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=114, channel_order='channels_first'):

        model = Sequential()

        model.add(Conv1D(64, 5, strides=2, input_shape=input_shape, data_format=channel_order, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(128, 5, strides=2, activation='relu', kernel_initializer='he_normal', data_format=channel_order))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(256, 5, strides=2, activation='relu', kernel_initializer='he_normal', data_format=channel_order))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(number_of_classes, activation='softmax', kernel_initializer='he_normal'))

        sgd = SGD(lr=0.009, momentum=0.9, decay=1e-10, nesterov=True)
        adam = Adam(lr=0.001, beta_1=0.5, beta_2=0.999)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model


class TBBModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=100, channel_order='channels_first'):

        model = Sequential()

        model.add(Conv1D(64, 3, input_shape=input_shape, data_format=channel_order, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(128, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Conv1D(256, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(number_of_classes, activation='softmax'))

        sgd = SGD(lr=0.008, decay=1e-10, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])

        return model


class YHModel:
    @staticmethod
    def get_model(input_shape=None, number_of_classes=114, channel_order='channels_first'):

        model = Sequential()

        model.add(Conv1D(64, 3, padding='same', kernel_initializer='he_normal', data_format=channel_order, input_shape=input_shape))
        model.add(Activation("elu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, 3, padding='same', kernel_initializer='he_normal'))
        model.add(Activation("elu"))
        model.add(Conv1D(128, 3, padding='same', kernel_initializer='he_normal'))
        model.add(Activation("elu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(256, 3, padding='same', kernel_initializer='he_normal'))
        model.add(Activation("elu"))
        model.add(Conv1D(256, 3, padding='same', kernel_initializer='he_normal'))
        model.add(Activation("elu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation("elu"))
        model.add(Dropout(0.5))

        model.add(Dense(512, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation("elu"))
        model.add(Dropout(0.5))

        model.add(Dense(256, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation("elu"))
        model.add(Dropout(0.5))

        model.add(Dense(number_of_classes))
        model.add(Activation("softmax"))

        opt = SGD(lr=0.01, momentum=0.9, decay=0.01 / 40, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', fmeasure, precision, recall])

        return model
