"""
激活函数库
"""
import paddle
import matplotlib.pyplot as plt


# 显示图像
def plt_fn(fig1, fig2):
    x = paddle.linspace(-10, 10, 10000)
    plt.figure()         # 创建新图像
    plt.plot(x.tolist(), fig1(x).tolist(), color='#40E0D0', label=f'{fig1.__name__} function')
    plt.plot(x.tolist(), fig2(x).tolist(), color='#4169E1', linestyle='--', label=f'{fig2.__name__} function')
    ax = plt.gca()       # 获取当前坐标轴对象
    ax.spines['top'].set_color('none')       # 取消上侧坐标轴
    ax.spines['right'].set_color('none')     # 取消右侧坐标轴
    ax.xaxis.set_ticks_position('bottom')    # 设置默认的x坐标轴
    ax.yaxis.set_ticks_position('left')      # 设置默认的y坐标轴
    ax.spines['left'].set_position(('data', 0))     # 设置y轴起点为0
    ax.spines['bottom'].set_position(('data', 0))   # 设置x轴起点为0
    plt.legend(loc='lower right', fontsize='medium')    # 图例
    plt.savefig(f'T4.1-{fig1.__name__}和{fig2.__name__}函数.jpg')     # 保存图片
    plt.show()      # 显示图像


# sigmoid型函数
# -logistic函数
def logistic(x):
    return 1 / (1 + paddle.exp(-x))


# -tanh函数
def tanh(x):
    return (paddle.exp(x) - paddle.exp(-x)) / (paddle.exp(x) + paddle.exp(-x))


# if __name__ == '__main__':
#     x = paddle.linspace(-10, 10, 10000)
#     plt.figure()         # 创建新图像
#     plt.plot(x.tolist(), logistic(x).tolist(), color='#40E0D0', label='logistic function')
#     plt.plot(x.tolist(), tanh(x).tolist(), color='#4169E1', linestyle='--', label='tanh function')
#     ax = plt.gca()       # 获取当前坐标轴对象
#     ax.spines['top'].set_color('none')       # 取消上侧坐标轴
#     ax.spines['right'].set_color('none')     # 取消右侧坐标轴
#     ax.xaxis.set_ticks_position('bottom')    # 设置默认的x坐标轴
#     ax.yaxis.set_ticks_position('left')      # 设置默认的y坐标轴
#     ax.spines['left'].set_position(('data', 0))     # 设置y轴起点为0
#     ax.spines['bottom'].set_position(('data', 0))   # 设置x轴起点为0
#     plt.legend(loc='lower right', fontsize='large')    # 图例
#     plt.savefig('T4.1-Logistic和tanh函数.jpg')     # 保存图片
#     plt.show()      # 显示图像


# -Hard_Logistic函数
def hard_logistic(x):
    return paddle.maximum(paddle.minimum((paddle.to_tensor(0.25 * x + 0.5)),
                                         paddle.to_tensor(1.)), paddle.to_tensor(0.))


# -Hard_Tanh函数
def hard_tanh(x):
    return paddle.maximum(paddle.minimum(x, paddle.to_tensor(1.)), paddle.to_tensor(-1.))


if __name__ == '__main__':
    plt_fn(hard_logistic, logistic)


# 斜坡型函数
# -ReLU函数
def relu(x):
    return paddle.maximum(x, paddle.to_tensor(0.))


# -带泄露的ReLU函数
def leaky_relu(x, negative_slope=0.1):
    a1 = paddle.cast((x > 0), dtype='float32') * x
    a2 = paddle.cast((x <= 0), dtype='float32') * negative_slope * x
    return a1 + a2


# if __name__ == '__main__':
#     x = paddle.linspace(-10, 10, 10000)
#     plt.figure()         # 创建新图像
#     plt.plot(x.tolist(), relu(x).tolist(), color='r', label='ReLU function')
#     plt.plot(x.tolist(), leaky_relu(x).tolist(), color='r', linestyle='--', label='Leaky_ReLU function')
#     ax = plt.gca()       # 获取当前坐标轴对象
#     ax.spines['top'].set_color('none')       # 取消上侧坐标轴
#     ax.spines['right'].set_color('none')     # 取消右侧坐标轴
#     ax.xaxis.set_ticks_position('bottom')    # 设置默认的x坐标轴
#     ax.yaxis.set_ticks_position('left')      # 设置默认的y坐标轴
#     ax.spines['left'].set_position(('data', 0))     # 设置y轴起点为0
#     ax.spines['bottom'].set_position(('data', 0))   # 设置x轴起点为0
#     plt.legend(loc='upper left', fontsize='medium')    # 图例
#     plt.savefig('T4.1-ReLU和Leaky_ReLU函数.jpg')     # 保存图片
#     plt.show()      # 显示图像


# -ELU函数


# -Softplus函数


# -Swish函数


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