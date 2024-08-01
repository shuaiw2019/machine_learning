from Bridge import oprator
from Bridge import dataset
import numpy as np


# 线性函数
def linear_func(x, w = 1.2, b = 0.5):
    y = x * w + b
    return y


from matplotlib import pyplot as plt

func = linear_func
interval = (-10,10)
train_num = 100 # 训练样本数目
test_num = 50 # 测试样本数目
noise = 2
X_train, y_train = dataset.create_toy_data(func=func, interval=interval, sample_num=train_num, noise = noise, add_outlier = False)
X_test, y_test = dataset.create_toy_data(func=func, interval=interval, sample_num=test_num, noise = noise, add_outlier = False)

# X_train_large, y_train_large = dataset.create_toy_data(func=func, interval=interval, sample_num=5000, noise = noise, add_outlier = False)

# 生成数据用于绘制直线
# paddle.linspace返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num
X_underlying = np.linspace(interval[0], interval[1], train_num)
y_underlying = linear_func(X_underlying)

# # 绘制数据
# plt.scatter(X_train, y_train, facecolor='none', edgecolors='b', s=50, marker='o', label='train data')
# plt.scatter(X_test, y_test, facecolor='none', edgecolors='r', s=50, marker='*', label='train data')
# plt.plot(X_underlying, y_underlying, c='g', label='underlying distribution')
# plt.legend()
# plt.savefig('线性回归训练数据')
# plt.show()


# 模型训练，100份数据
input_size = 1
reg_lambda = 0
model = oprator.linear(input_size)

# model = oprator.optimizer_lsm(model, X_train.reshape([-1, 1]), y_train.reshape([-1, 1]), reg_lambda=reg_lambda)
model=model
X=X_train.reshape([-1, 1])
y=y_train.reshape([-1, 1])
reg_lambda=0

N, D = X.shape
# 对输入特征数据所有特征向量求平均
x_bar_tran = np.mean(X, axis=0).T

# 求标签的平均值，shape=[1]
y_bar = np.mean(y)

# np.subtract通过广播的方式实现矩阵减向量
x_sub = np.subtract(X, x_bar_tran)

# np.inverse求方阵的逆
tmp = np.linalg.inv(np.matmul(x_sub.T, x_sub) + reg_lambda * np.eye(D))


w = np.matmul(np.matmul(tmp, x_sub.T), (y - y_bar))
b = y_bar - np.matmul(x_bar_tran, w)
print('y:',y.shape)

print('n:',N,'\nd:',D)


# print('w_pred:', model.params['w'].item(), 'b_pred:', model.params['b'].item())
#
# y_train_pred = model(X_train.reshape([-1, 1])).squeeze()
# train_error = oprator.mse(y_true=y_train,y_pred=y_train_pred).item()
# print('train error:', train_error)
#
# # 模型评价，50份数据
# y_test_prd = model(X_test.reshape([-1, 1])).squeeze()
# test_error = oprator.mse(y_true=y_test, y_pred=y_test_prd).item()
# print('test error:', test_error)