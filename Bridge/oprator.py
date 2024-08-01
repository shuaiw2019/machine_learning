"""
自定义算子
"""
import numpy as np


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
class add(Op):
    def __init__(self):
        super(add, self).__init__()

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
class multiply(Op):
    def __init__(self):
        super(multiply, self).__init__()

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
import math
class exponential(Op):
    def __init__(self):
        super(exponential, self).__init__()

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
# 设置随机种子
np.random.seed(1)
class linear(Op):
    def __init__(self, input_size):
        """
        构造函数，生成模型参数
        :param input_size: 模型要处理的数据特征向量长度
        """
        super().__init__()
        self.input_size = input_size
        # 模型参数
        self.params = {}
        self.params['w'] = np.random.randn(self.input_size, 1)
        self.params['b'] = np.zeros(shape=[1], dtype='float32')

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向计算数据的预测值
        :param X: 张量，shape=[N,D]
        :return:  y_pred:张量，shape=[N]
        """
        N,D = X.shape
        if self.input_size == 0:
            return np.full(shape=[N,1], fill_value=self.params['b'])
        # 输入数据维度合法性验证
        assert D == self.input_size
        # 使用np.tensordot计算两个tensor的乘积
        # y_pred = np.tensordot(X, self.params['w'], axes=1) + self.params['b']
        y_pred = np.dot(X, self.params['w']) + self.params['b']
        return y_pred


if __name__ == '__main__':
    input_size = 3
    N = 2
    X = np.random.randn(N, input_size)
    model = linear(input_size)
    y_pred = model(X)
    print('y_pred:', y_pred)
    print(type(y_pred))
    print(y_pred.shape)


# 均方误差（Mean Squared Error, MSE）
def mse(y_true, y_pred):
    """
    计算标签真实值y_true和预测值 y_pred之间的均方误差
    :param y_true: 样本真实标签
    :param y_pred: 样本预测标签
    :return: error: 误差值
    """
    # 输入数据维度合法性验证
    assert y_true.shape[0] == y_pred.shape[0]
    # 计算均方误差
    error = np.mean(np.square(y_true - y_pred))
    return error


if __name__ == '__main__':
    y_true = np.array([[-0.2],[4.9]], dtype='float32')
    y_pred = np.array([[1.3],[2.5]], dtype='float32')
    error = mse(y_true=y_true,y_pred=y_pred).item()
    print('error:', error)


# 优化器
def optimizer_lsm(model, X, y, reg_lambda=0):
    """
    最小二乘法求解线性回归模型的解析解
    :param model: 模型
    :param X: 张量，特征数据，shape=[N,D]
    :param y: 张量，标签数据，shape=[N]
    :param reg_lambda: float，正则化系数
    :return: model: 优化好的模型
    """
    N, D = X.shape
    # 对输入特征数据所有特征向量求平均
    x_bar_tran = np.mean(X, axis=0).T
    # 求标签的平均值，shape=[1]
    y_bar = np.mean(y)
    # np.subtract通过广播的方式实现矩阵减向量
    x_sub = np.subtract(X, x_bar_tran)

    # 使用np.all判断输入张量是否全是0
    if np.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = np.zeros(shape=[D])
        return model

    # np.linalg.inv求方阵的逆
    tmp = np.linalg.inv(np.matmul(x_sub.T, x_sub) + reg_lambda * np.eye(D))

    w = np.matmul(np.matmul(tmp, x_sub.T), (y - y_bar))
    b = y_bar - np.matmul(x_bar_tran, w)

    model.params['w'] = np.squeeze(w, axis=-1)
    model.params['b'] = b

    return model


# 多项式变换
def polynomial_basic_function(x, degree=2):
    """
    对原始特征x进行变换，增加特征
    :param x: 张量，输入数据，shape=[N,1]
    :param degree: int，多项式阶数
    :return: 张量，数据变换结果
    """
    if degree == 0:
        return np.ones(shape=x.shape, dtype='float32')

    x_tmp = x
    x_result = x_tmp

    for i in range(2, degree + 1):
        x_tmp = np.multiply(x_tmp, x) # 逐元素相乘
        x_result = np.concatenate((x_result, x_tmp), axis=-1)

    return x_result


if __name__ == '__main__':
    data = [[2], [3], [4]]
    X = np.array(data, dtype='float32')
    degree = 3
    trans_X = polynomial_basic_function(X, degree=degree)
    print('转换前：\n', X)
    print('阶数为',degree,'转换后：\n',trans_X)


#