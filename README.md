# Python-Mnist

## 仓库简介
- **该仓库是基于手写数字识别mnist数据集**
- **"python神经网络编程"-第二章**
- **李宏毅2017年机器学习kerasDNN-Demo**
- **自己搭建了一个卷积神经网络CNN-Demo**


## 训练步骤
- [Keras中文文档](https://keras.io/zh/getting-started/sequential-model-guide/)

### 1.普通三层network
**首先实现"python神经网络编程"这本书的第二章的内容，自己手动搭建神经网络，不使用任何框架来实现。**

**这是一个三层神经网络：输入层、隐藏层、输出层。**

**训练过程：**

```
def train(self, inputs_list, targets_list):
        """----------计算正向传播各层实际输出----------"""

        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入标签转化成二维矩阵
        targets = np.array(targets_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        """----------计算误差----------"""

        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐层误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新隐层与输出层之间的权重 a*Ek*Ok*(1-Ok)*Oj
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # 更新隐层与输出层之间的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
```


### 2.DNN神经网络

**在第一步的基础上，我们使用keras框架来实现 fully connected 搭建神经网络**

**这一步，主要借鉴了李宏毅老师在2017年公开课当中的keras内容**

```
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
```

##### 注意：
>在训练数据之前要对数据进行预处理
>
```
# 修改shape
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
```


### 3.CNN神经网络
**卷积神经网络不是fully connected 有卷积层、池化层。在keras的代码中就是修改相应的add函数。**

**训练CNN网络前也需要对数据进行预处理。**


**卷积层数、池化层数都是自定义的没有特定的要求**
