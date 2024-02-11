from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

class MnistCnn(Sequential):
    def __init__(self, layers=None, name=None):
        super().__init__(layers, name)
        self.add(Conv2D(30, 2, activation= 'relu', padding='same', input_shape = (28,28,1)))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Conv2D(30, 2, activation= 'relu', padding='same'))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Conv2D(30, 2, activation= 'relu', padding='same'))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Conv2D(30, 2, activation= 'relu', padding='same'))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Flatten())
        self.add(Dense(10, activation="softmax"))
