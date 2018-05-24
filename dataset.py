from keras.datasets import cifar10
import numpy as np

from utils import one_hot

NUM_CLASSES = 10

class Cifar(object):
    def __init__(self, images, labels):
        self._i = 0
        self.images = images
        self.labels = labels

    def next_batch(self, batch_size):
        batch_images = self.images[self._i: self._i + batch_size]
        batch_labels = self.labels[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return batch_images, batch_labels

class CifarDataManager(object):

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize
        x_train = x_train.astype(float) / 255
        x_test = x_test.astype(float) / 255

        # one_hot
        y_train = one_hot(y_train, NUM_CLASSES)
        y_test = one_hot(y_test, NUM_CLASSES)

        self.train = Cifar(images=x_train, labels=y_train)
        self.test = Cifar(images=x_test, labels=y_test)
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)
        return self

