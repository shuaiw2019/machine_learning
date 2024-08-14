"""
算子库
"""
import matplotlib.pyplot as plt
import numpy as np
import paddle
import math
from project01.nndl import activation


# 算子Op接口
class Op(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    # 前向函数
    # 输入：张量inputs
    # 输出：张量outputs
    def forward(self, *args):
        # return outputs
        raise NotImplementedError

    # 反向函数
    # 输入：最终输出对outputs的梯度outputs_grads
    # 输出：最终输出对inputs的梯度inputs_grads
    def backward(self, outputs_grads):
        # return inputs_grads
        raise NotImplementedError


# 加法算子
class Add(Op):
    def __init__(self):
        super(Add, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x + y
        return outputs

    def backward(self, grads):
        grads_x = grads * 1
        grads_y = grads * 1
        return grads_x, grads_y


# 乘法算子
class Multiply(Op):
    def __init__(self):
        super(Multiply, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x * y
        return outputs

    def backward(self, grads):
        grads_x = grads * self.y
        grads_y = grads * self.x
        return grads_x, grads_y


# 指数算子
class Exponential(Op):
    def __init__(self):
        super(Exponential, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        outputs = math.exp(self.x)
        return outputs

    def backward(self, grads):
        grads = grads * math.exp(self.x)
        return grads


# 向量线性乘法算子
class Linear(Op):
    def __init__(self, input_size, output_size, name, weight_init=paddle.standard_normal,bias_init=paddle.zeros):
        """
           构造广义线性函数，初始化模型参数
        Args:
            input_size: 模型要处理的数据特征向量长度，或神经元数量
            output_size: 输出数据维度，或输出神经元数量
            name: 算子名称
            weight_init: 权重初始化
            bias_init: 偏置初始化
        """
        super().__init__()
        # 模型参数
        self.params = {}
        self.params['w'] = weight_init(shape=[input_size, output_size])
        self.params['b'] = bias_init(shape=[1, output_size])
        self.inputs = None
        self.name = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        param:
          inputs: 输入，shape=[N,input_size]，N是样本数量
        return:
          outputs: 输出，shape=[N,output_size]
        """
        self.inputs = inputs
        # 使用paddle.matmul计算两个tensor的乘积
        outputs = paddle.matmul(inputs, self.params['w']) + self.params['b']

        return outputs


if __name__ == '__main__':
    # 设置随机种子
    paddle.seed(10)
    input_size = 3
    N = 2
    X = paddle.randn(shape=[N, input_size],dtype='float32')
    model = Linear(input_size=input_size, output_size=1, name=None)
    y_pred = model(X)
    print(X)
    print('y_pred:', y_pred,'\ny_pred type:', type(y_pred),'\ny_pred shape:',y_pred.shape)


# logistic回归算子
class ModelLR(Op):
    def __init__(self, input_dim):
        super(ModelLR, self).__init__()
        self.params = {}
        self.params['w'] = paddle.zeros(shape=[input_dim, 1])
        self.params['b'] = paddle.zeros(shape=[1])
        self.grads = {}
        self.X = None
        self.outputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        inputs: shape=[N,D],N是样本数量，D是特征数量
        outputs: 预测标签为1的概率，shape=[N,1]
        """
        self.X = inputs
        # 线性计算
        score = paddle.matmul(inputs, self.params['w']) + self.params['b']
        self.outputs = activation.logistic(score)
        return self.outputs

    def backward(self, labels):
        """
        labels: 真实标签，shape=[N,1]
        """
        N = labels.shape[0]
        self.grads['w'] = -1 / N * paddle.matmul(self.X.t(), (labels - self.outputs))
        self.grads['b'] = -1 / N * paddle.sum(labels - self.outputs)


if __name__ == '__main__':
    # 固定随机种子，保持每次运行结果一致
    paddle.seed(0)
    # 随机生成3条长度为4的数据
    inputs = paddle.randn(shape=[3, 4])
    print('Input is:', inputs)
    # 实例化模型
    model = ModelLR(4)
    outputs = model(inputs)
    print('Output is:', outputs)


# softmax回归算子
class ModelSR(Op):
    def __init__(self, input_dim, output_dim):
        super(ModelSR, self).__init__()
        self.params = {}
        self.params['w'] = paddle.zeros(shape=[input_dim, output_dim])
        self.params['b'] = paddle.zeros(shape=[output_dim])
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        softmax函数计算
        Args:
            inputs: shape=[N,D]，N是样本数量，D是特征维度
        Returns:
            outputs: 预测值，shape=[N,C],C是类别数
        """
        self.X = inputs
        # 线性计算
        score = paddle.matmul(self.X, self.params['w']) + self.params['b']
        self.outputs = activation.softmax(score)
        return self.outputs

    def backward(self, labels):
        """
        计算梯度
        Args:
            labels: 样本真实标签，shape=[N,C]
        """
        N = labels.shape[0]
        # 转换为one-hot向量
        labels = paddle.nn.functional.one_hot(labels, self.output_dim)
        self.grads['w'] = - 1 / N * paddle.matmul(self.X.t(), (labels - self.outputs))
        self.grads['b'] = - 1 / N * paddle.sum(labels - self.outputs)


if __name__ == '__main__':
    inputs = paddle.randn(shape=[1, 4])
    print('input is:', inputs)
    model = ModelSR(input_dim=4, output_dim=3)
    outputs = model(inputs=inputs)
    print('output is:', outputs)


# 感知器分类算子
class Perceptron(Op):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.params = {}
        self.params['w'] = paddle.zeros(shape=[input_dim, output_dim])
        self.params['b'] = paddle.zeros(shape=[output_dim])
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        感知器计算
        Args:
            inputs: shape=[N,D],N是样本数量，D是特征维度
        Returns:
            outputs: 预测值，shape=[N,C],C是类别数
        """
        self.X = inputs
        score = paddle.matmul(self.X, self.params['w']) + self.params['b']
        self.outputs = activation.sign(score)
        return self.outputs

    def backward(self, labels):
        """
        梯度计算
        Args:
            labels: 样本真实标签，shape=[N,C]
        """
        N = labels.shape[0]
        labels = paddle.nn.functional.one_hot(labels, self.output_dim)


# 损失函数
# 二分类交叉熵损失函数
class BinaryCrossEntropyLoss(Op):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.num = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: 预测值，shape=[N,1]
        labels: 真实标签，shape=[N,1]
        返回：损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = -1. / self.num * (paddle.matmul(self.labels.t(), paddle.log(self.predicts))
                                 + paddle.matmul((1 - self.labels.t()), paddle.log(1 - self.predicts)))
        loss = paddle.squeeze(loss, axis=1)
        return loss


if __name__ == '__main__':
    labels = paddle.ones(shape=[3, 1])
    inputs = paddle.randn(shape=[3, 4])
    model = ModelLR(4)
    predicts = model(inputs)
    bce_loss = BinaryCrossEntropyLoss()
    print(bce_loss(predicts, labels))


# 多分类交叉熵损失函数
class MultiCrossEntroLoss(Op):
    def __init__(self):
        super(MultiCrossEntroLoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.num = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        多分类交叉熵的前向计算
        :param predicts: 预测值，shape=[N,1],N为样本数量
        :param labels: 真实标签，shape=[N,1]
        :return:平均损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = 0
        for i in range(0, self.num):
            index = self.labels[i]
            loss -= paddle.log(self.predicts[i][index])
        return loss / self.num


if __name__ == '__main__':
    inputs = paddle.randn(shape=[1, 4])
    model = ModelSR(input_dim=4, output_dim=3)
    outputs = model(inputs=inputs)
    labels = paddle.to_tensor([0])
    mce_loss = MultiCrossEntroLoss()

    print(mce_loss(outputs, labels).shape)


# 均方误差（Mean Squared Error, MSE）
class MeanSquaredError(Op):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        """
            计算标签真实值y_true和预测值 y_pred之间的均方误差
            :param y_true: 样本真实标签
            :param y_pred: 样本预测标签
            :return: error: 误差值
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # 输入数据维度合法性验证
        assert self.y_true.shape[0] == self.y_pred.shape[0]
        # 计算均方误差
        error = paddle.mean(paddle.square(self.y_true - self.y_pred))

        return error


if __name__ == '__main__':
    y_true = paddle.to_tensor([[-0.2],[4.9]], dtype='float32')
    y_pred = paddle.to_tensor([[1.3],[2.5]], dtype='float32')
    error = MeanSquaredError()
    print('error:', error(y_true, y_pred))