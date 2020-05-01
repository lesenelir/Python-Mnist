from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, adam

import matplotlib.pyplot as plt
import numpy as np

"""
2017李宏毅 全连接网络
"""


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]  # 第0行取前第0~9999列
    y_train = y_train[0:number]

    # 转换维度为一维
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    # 设置成小数，方便收敛
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 类标签 归类
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test

    # 归一化
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

# print(x_train.shape)  # (10000, 784)
# print(y_train.shape)  # (10000, 10)
# print(x_test.shape)  # (10000, 784)
# print(y_test.shape)  # (10000, 10)

# 创建神经网络
model = Sequential()
model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))  # 增加第一层输入层，维度是28*28，后面连接的第一层隐藏层的神经元个数是689，激活函数是relu
model.add(Dropout(0.7))
model.add(Dense(units=689, activation='relu'))  # 第二层隐藏层神经元个数689，激活函数是relu
model.add(Dropout(0.7))
model.add(Dense(units=10, activation='softmax'))
# 对模型进行设置，设置loss function，learning rate的优化函数，metrics用于设定评估当前训练模型的性能的评估函数

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练 batch_size 不能过快
model.fit(x_train, y_train, batch_size=100, epochs=20)

score = model.evaluate(x_train, y_train)
print('\nTrain Acc:', score[1])  # Train Acc: 0.992

result = model.evaluate(x_test, y_test)
print('\nTest Acc:', result[1])  # Test Acc: 0.9603

# 拉取test图
test_run = x_test[9999].reshape(1, 28*28)
test_label = y_test[9999]
print('label:--->', test_label)
plt.imshow(test_run.reshape([28 * 28]))



