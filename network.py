import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class neuralNetwork:

    # 用于神经网络初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入层节点数
        self.inodes = inputnodes
        # 隐层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 初始化输入层与隐层之间的权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐层与输出层之间的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数用匿名函数来实现（S函数）scipy.special.expit
        self.activation_function = lambda x: scipy.special.expit(x)

    # 神经网络学习训练
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

    # 神经网络测试 正向传播
    def test(self, inputs_list):  # 读取图片 扁平化
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == "__main__":
    # 初始化 784（28 * 28）个输入节点，200个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.3 优化后0.1
    learning_rate = 0.1
    # 世代-训练次数-神经网络层数
    epochs = 5
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取训练集
    training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 训练数据
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            # 输入数据范围（0.01~1）
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            # 标记数据（相应标记为0.99，其余0.01）
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # 读取测试数据
    test_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 打印测试数据标签
    test_data = test_data_list[0].split(',')
    print('原标签：', test_data[0])

    # 生成标签图片
    image_array = np.asfarray(test_data[1:]).reshape(28, 28)
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    # 利用神经网络预测
    results = n.test(np.asfarray(test_data[1:]) / 255.0 * 0.99 + 0.01)  # 转换np矩阵
    pre_label = np.argmax(results)
    print('预测结果：', pre_label)
    print(results)
