import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# setup data shape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 设置成小数，方便收敛
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train = x_train / 255
x_test = x_test / 255

# 归类 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
x_train = x_train
x_test = x_test

# 建立网络
model = Sequential()

# 1st layer
model.add(Convolution2D(
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    input_shape=(28, 28, 1)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same"
))

# 2nd layer
model.add(Convolution2D(
    filters=64,
    kernel_size=[5, 5],
    padding="same"
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same"
))

# 1st fully connected dense
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

# 2nd fully connected dense
model.add(Dense(10))
model.add(Activation('softmax'))

# Define optimizer and param
# adam = Adam(lr=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, batch_size=100)

model.save_weights('./CNN_Own_Mnist.h5', overwrite=True)

score = model.evaluate(x_train, y_train)
print('\nTrain Acc:', score[1])  # Train Acc: 0.9977

result = model.evaluate(x_test, y_test)
print('\nTest Acc:', result[1])  # Test Acc: 0.9912


