from MyClasses.CnnClasses import MnistCnn
from keras.datasets import mnist
from matplotlib.pyplot import figure, show, plot, legend, subplots
from numpy import zeros, arange

# Configuration
load_model = False
model_path = './Weights/mnist_cnn'
num_epochs = 30

# Flow
model = MnistCnn()
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train_mat = zeros((x_train.shape[0],10))
y_test_mat = zeros((x_test.shape[0],10))

y_train_mat[arange(x_train.shape[0]), y_train] = 1
y_test_mat[arange(x_test.shape[0]), y_test] = 1

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

if load_model:
    model.load_weights(model_path)

train_accuracy = []
test_accuracy = []
previous_test_acc = 0

for ii in range(num_epochs):
    print(f"\nEpoch {ii}")
    history = model.fit(x_train, y_train_mat, 
                        epochs=1,
                        batch_size=128,
                        validation_data=(x_test, y_test_mat))
    test_loss, test_acc = model.evaluate(x_test,  y_test_mat)
    train_accuracy.append(history.history['accuracy'])
    test_accuracy.append(test_acc)
    if test_acc > previous_test_acc:
        previous_test_acc = test_acc
        model.save_weights(f'./Weights/mnist_cnn_{test_acc}')

fig,ax = subplots()
ax.plot(train_accuracy, label="Train accuracy")
ax.plot(test_accuracy, label="Test accuracy")
ax.legend()
show()
