"""
激活函数库
"""
import paddle


# logistic函数
def logistic(x):
    return 1 / (1 + paddle.exp(-x))


# if __name__ == '__main__':
#     x = paddle.linspace(-10, 10, 10000)
#     plt.figure()         # 创建新图像
#     plt.plot(x.tolist(), logistic(x).tolist(), color='r', label='logistic function')
#     ax = plt.gca()       # 获取当前坐标轴对象
#     ax.spines['top'].set_color('none')       # 取消上侧坐标轴
#     ax.spines['right'].set_color('none')     # 取消右侧坐标轴
#     ax.xaxis.set_ticks_position('bottom')    # 设置默认的x坐标轴
#     ax.yaxis.set_ticks_position('left')      # 设置默认的y坐标轴
#     ax.spines['left'].set_position(('data', 0))     # 设置y轴起点为0
#     ax.spines['bottom'].set_position(('data', 0))   # 设置x轴起点为0
#     plt.legend()    # 图例
#     plt.savefig('T3.1-Logistic函数.jpg')     # 保存图片
#     plt.show()      # 显示图像


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


# 观察softmax的计算方式
if __name__ == '__main__':
    X = paddle.to_tensor([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]])
    predict = softmax(X)
    print('softmax函数结果：', predict)


# 符号函数
def sign(X):
    return paddle.sign(X)


if __name__ == '__main__':
    X =paddle.to_tensor([[2.0, -3.0, 3.0], [-3.0, 4.0, -4.0]])
    print('符号函数结果：', sign(X))