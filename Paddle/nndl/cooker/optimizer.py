from abc import abstractmethod
import paddle


# 优化器基类
class Optimizer(object):
    def __init__(self, init_lr, model):
        """
        优化器类初始化
        init_lr: 初始化学习率
        model： 需要优化的模型
        """
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        """
        定义每次迭代如何更新参数
        """
        pass


# 采用梯度下降法的优化器
class SimpleBatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(SimpleBatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        """
        更新参数
        """
        # 遍历所有参数，并更新参数
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] = self.model.params[key] - self.model.grads[key]


# 最小二乘法计算解析解
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
    x_bar_tran = paddle.mean(X, axis=0).T
    # 求标签的平均值，shape=[1]
    y_bar = paddle.mean(y)
    # paddle.subtract通过广播的方式实现矩阵减向量
    x_sub = paddle.subtract(X, x_bar_tran)

    # 使用paddle.all判断输入张量是否全是0
    if paddle.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = paddle.zeros(shape=[D])
        return model

    # paddle.inverse求方阵的逆
    tmp = paddle.inverse(paddle.matmul(x_sub.T, x_sub) + reg_lambda * paddle.eye(num_rows=(D)))

    w = paddle.matmul(paddle.matmul(tmp, x_sub.T), (y - y_bar))
    b = y_bar - paddle.matmul(x_bar_tran, w)

    model.params['w'] = paddle.squeeze(w, axis=-1)
    model.params['b'] = b

    return model