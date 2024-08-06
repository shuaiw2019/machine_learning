import paddle
from matplotlib import pyplot as plt
from Bridge.dataset import *

paddle.seed(0)
n_samples = 1000
X, y = make_multiclass(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

# # 可视化数据集
# plt.figure(figsize=(5, 5))
# plt.scatter(x=X[:,0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
# plt.savefig('T3.2-多分类数据.jpg')
# plt.show()

num_train = 640
num_dev = 160
num_test = 200

X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[-num_test:], y[-num_test:]
# 查看数据维度
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_dev:', X_dev.shape, 'y_dev:', y_dev.shape)
print('X_test:', X_test, 'y_test:', y_test)


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


#