"""
激活函数库
"""
import paddle
import matplotlib.pyplot as plt


# 显示图像
def plt_fn(*functions_with_args):
    x = paddle.linspace(-10, 10, 10000)
    plt.figure()  # 创建新图像
    for i, (func, args) in enumerate(functions_with_args):
        # 使用不同的颜色和线型
        color = f"C{i}"  # Matplotlib默认颜色循环
        linestyle = '-' if i % 2 == 0 else '--'  # 偶数索引使用实线，奇数索引使用虚线
        y = func(x, *args)   # 传入参数
        plt.plot(x.tolist(), y.tolist(), color=color, linestyle=linestyle, label=f'{func.__name__} function')
    ax = plt.gca()       # 获取当前坐标轴对象
    ax.spines['top'].set_color('none')       # 取消上侧坐标轴
    ax.spines['right'].set_color('none')     # 取消右侧坐标轴
    ax.xaxis.set_ticks_position('bottom')    # 设置默认的x坐标轴
    ax.yaxis.set_ticks_position('left')      # 设置默认的y坐标轴
    ax.spines['left'].set_position(('data', 0))     # 设置y轴起点为0
    ax.spines['bottom'].set_position(('data', 0))   # 设置x轴起点为0
    plt.legend(loc=0, fontsize='medium')    # 图例
    plt.savefig(f'T4.1-{"、".join(func.__name__ for func, _ in functions_with_args)}函数.jpg')     # 保存图片
    plt.show()      # 显示图像


# sigmoid型函数
# -logistic函数
def logistic(x):
    return 1 / (1 + paddle.exp(-x))


# -tanh函数
def tanh(x):
    return (paddle.exp(x) - paddle.exp(-x)) / (paddle.exp(x) + paddle.exp(-x))


# -Hard_Logistic函数
def hard_logistic(x):
    return paddle.maximum(paddle.minimum((paddle.to_tensor(0.25 * x + 0.5)),
                                         paddle.to_tensor(1.)), paddle.to_tensor(0.))


# -Hard_Tanh函数
def hard_tanh(x):
    return paddle.maximum(paddle.minimum(x, paddle.to_tensor(1.)), paddle.to_tensor(-1.))


# if __name__ == '__main__':
#     func1 = [(hard_logistic, ()), (logistic, ())]
#     func2 = [(hard_tanh, ()), (tanh, ())]
#     plt_fn(*func1)
#     plt_fn(*func2)


# 斜坡型函数
# -ReLU函数
def relu(x):
    return paddle.maximum(x, paddle.to_tensor(0.))


# -带泄露的ReLU函数
def leaky_relu(x, negative_slope=0.1):
    a1 = paddle.cast((x > 0), dtype='float32') * x
    a2 = paddle.cast((x <= 0), dtype='float32') * negative_slope * x
    return a1 + a2


# -ELU函数
def elu(x, gamma=1):
    return relu(x) + paddle.minimum(gamma * (paddle.exp(x) - paddle.to_tensor(1.)), paddle.to_tensor(0.))


# -Softplus函数
def softplus(x):
    return paddle.log(paddle.to_tensor(1.) + paddle.exp(x))


# if __name__ == '__main__':
#     func = [(relu, ()), (leaky_relu, ()), (elu, ()), (softplus, ())]
#     plt_fn(*func)


# -Swish函数
def swish(x, beta=1):
    return x * logistic(beta * x)


# if __name__ == '__main__':
#     betas = [0, 0.5, 1, 100]
#     functions_with_args = [(swish, (beta,)) for beta in betas]  # 列表推导式
#     plt_fn(*functions_with_args)


# 定义Softmax函数
def softmax(X):
    """
    Softmax函数
    Args:
        X: shape=[N,C],N为向量数量，C为向量维度
    Returns: 函数计算结果
    """
    x_max = paddle.max(X, axis=1, keepdim=True)
    x_exp = paddle.exp(X - x_max)
    partition = paddle.sum(x_exp, axis=1, keepdim=True)
    return x_exp / partition


# # 观察softmax的计算方式
# if __name__ == '__main__':
#     X = paddle.to_tensor([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]])
#     predict = softmax(X)
#     print('softmax函数结果：', predict)


# 符号函数
def sign(X):
    return paddle.sign(X)


# if __name__ == '__main__':
#     X =paddle.to_tensor([[2.0, -3.0, 3.0], [-3.0, 4.0, -4.0]])
#     print('符号函数结果：', sign(X))