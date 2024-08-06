"""
自定义函数，用于生成各种数据集
"""

# 生成用于线性回归的数据集
import paddle
import math
import numpy as np


def create_toy_data(func, interval, sample_num, noise = 0.0, add_outlier = False, outlier_ratio = 0.001):
    """
    根据给定的函数，生成样本
    输入：
       - func：函数
       - interval： x的取值范围
       - sample_num： 样本数目
       - noise： 噪声标准差
       - add_outlier：是否生成异常值
       - outlier_ratio：异常值占比
    输出：
       - X: 特征数据，shape=[n_samples,1]
       - y: 标签数据，shape=[n_samples,1]
    """
    # 均匀采样
    # 使用paddle.rand在生成sample_num个随机数
    X = paddle.rand(shape=[sample_num]) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用paddle.normal生成0均值，noise标准差的数据
    epsilon = paddle.normal(0, noise, paddle.to_tensor(y.shape[0],))
    y = y + epsilon
    # 生成额外的异常点
    if add_outlier:
        outlier_num = int(len(y) * outlier_ratio)
        if outlier_num != 0:
            # 使用paddle.randint生成服从均匀分布的、范围在[0, len(y))的随机Tensor
            outlier_idx = paddle.randint(0, len(y), shape=[outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y


def make_moons(n_samples=1000, shuffle=True, noise=None):
    """
    生成带噪音的弯月形状数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples]
    """
    n_samples_out = n_samples // 2    # 整除2，两类数据
    n_samples_in = n_samples - n_samples_out

    # 采集第1类数据，特征为(x,y)
    # 使用'paddle.linspace'在0到pi上均匀取n_samples_out个值
    # 使用'paddle.cos'计算上述取值的余弦值作为特征1，使用'paddle.sin'计算上述取值的正弦值作为特征2
    outer_circ_x = paddle.cos(paddle.linspace(0, math.pi, n_samples_out))
    outer_circ_y = paddle.sin(paddle.linspace(0, math.pi, n_samples_out))

    inner_circ_x = 1 - paddle.cos(paddle.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 0.5 - paddle.sin(paddle.linspace(0, math.pi, n_samples_in))

    print('outer_circ_x.shape:', outer_circ_x.shape, 'outer_circ_y.shape:', outer_circ_y.shape)
    print('inner_circ_x.shape:', inner_circ_x.shape, 'inner_circ_y.shape:', inner_circ_y.shape)

    # 使用'paddle.concat'将两类数据的特征1和特征2分别沿维度0拼接在一起，得到全部特征1和特征2
    # 使用'paddle.stack'将两类特征沿维度1堆叠在一起，堆叠后为[1000,2]
    X = paddle.stack(
        [paddle.concat([outer_circ_x, inner_circ_x]),
         paddle.concat([outer_circ_y, inner_circ_y])],
        axis=1
    )

    print('after concat shape:', paddle.concat([outer_circ_x, inner_circ_x]).shape)
    print('X shape:', X.shape)

    # 使用'paddle.zeros'将第一类数据的标签全部设置为0
    # 使用'paddle.ones'将第一类数据的标签全部设置为1
    y = paddle.concat(
        [paddle.zeros(shape=[n_samples_out]), paddle.ones(shape=[n_samples_in])]
    )

    print('y shape:', y.shape)

    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        # 使用'paddle.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor做索引值，用于打乱数据
        idx = paddle.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    # 如果noise不为None，则给特征值加入噪声
    if noise is not None:
        # 使用'paddle.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
        X += paddle.normal(mean=0.0, std=noise, shape=X.shape)

    return X, y


# 多分类数据集
def make_multiclass(n_samples=100, n_features=2, n_classes=3, shuffle=True, noise=0.1):
    """
    构件多分类数据集
    Args:
        n_samples: 数据量，int
        n_features: 特征数量，int
        n_classes: 类别数
        shuffle: 是否打乱数据， bool
        noise: 增加噪声的程度

    Returns:
        X: 特征数据，shape=[n_samples,n_features]
        y: 标签数据，shape=[n_samples,1]
    """
    # 计算每个类别的样本数量
    n_samples_per_class = [int(n_samples / n_classes) for k in range(n_classes)]
    for i in range(n_samples - sum(n_samples_per_class)):
        n_samples_per_class[i % n_classes] += 1
    # 将特征和标签初始化为0
    X = paddle.zeros([n_samples, n_features])
    y = paddle.zeros([n_samples], dtype='int32')
    # 随机生成3个簇中心作为类别中心
    centroids = paddle.randperm(2 ** n_features)[:n_classes]
    centroids_bin = np.unpackbits(centroids.numpy().astype('uint8')).reshape((-1, 8))[:, -n_features]
    centroids = paddle.to_tensor(centroids_bin, dtype='float32')
    # 控制簇中心的分离程度
    centroids = 1.5 * centroids - 1
    # 随机生成特征值
    X[:, :n_features] = paddle.randn(shape=[n_samples, n_features])

    stop = 0
    # 将每个类的特征值控制在簇中心附近
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_class[k]
        # 指定标签值
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        # 控制每个类别特征值的分散程度
        A = 2 * paddle.rand(shape=[n_features, n_features]) - 1
        X_k[...] = paddle.matmul(X_k, A)
        X_k += centroid
        X[start:stop, :n_features] = X_k

    # 如果noise不为None，则给特征加入噪声
    if noise > 0.0:
        # 生成noise掩膜，用来指定给哪些样本加入噪声
        noise_mask = paddle.rand([n_samples]) < noise
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                # 给加噪声的样本随机赋予标签值
                y[i] = paddle.randint(n_classes, shape=[1]).astype('int32')
    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        idx = paddle.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    return X, y

