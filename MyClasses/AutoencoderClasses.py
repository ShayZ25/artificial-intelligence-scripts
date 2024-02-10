from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D

class MnistAutoencoder(Sequential):
    def __init__(self, layers=None, name=None):
        super().__init__(layers, name)
        # Encoder
        self.add(Conv2D(6, 2, activation= 'relu', padding='same', input_shape = (28,28,1)))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Conv2D(12, 2, activation= 'relu', padding='same'))
        self.add(MaxPooling2D(2, padding= 'same'))
        self.add(Conv2D(24, 2, activation= 'relu', padding='same'))
        self.add(MaxPooling2D(2, padding= 'same'))

        # Decoder
        self.add(Conv2D(24, 2, activation= 'relu', padding='same'))
        self.add(UpSampling2D(2))
        self.add(Conv2D(12, 2, activation= 'relu', padding='same'))
        self.add(UpSampling2D(2))
        self.add(Conv2D(6,2,activation='relu', padding= 'same'))
        self.add(UpSampling2D(2))
        self.add(Conv2D(1,2,activation='relu', padding= 'same'))
        self.add(Cropping2D(((2,2),(2,2))))

