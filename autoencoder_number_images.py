from MyClasses.AutoencoderClasses import MnistAutoencoder
from keras.datasets import mnist
from matplotlib.pyplot import imshow, figure, subplot, gray, show

model = MnistAutoencoder()
model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

history = model.fit(x_train, x_train, 
                    epochs=11,
                    batch_size=128,
                    validation_data=(x_test, x_test))

pred = model.predict(x_test)

figure(figsize=(20, 4))
for i in range(5):
    # Display original
    ax = subplot(2, 5, i + 1)
    imshow(x_test[i].reshape(28, 28))
    gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = subplot(2, 5, i + 1 + 5)
    imshow(pred[i].reshape(28, 28))
    gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
show()
