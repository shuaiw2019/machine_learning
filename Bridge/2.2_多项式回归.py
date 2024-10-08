import matplotlib.pyplot as plt

from Bridge.oprator import *
from Bridge.dataset import *

import math


# sin函数：sin(2*pi*x)
def sin(x):
    y = np.sin(2 * math.pi * x)
    return y


# Toysin25训练集
# 生成数据
func = sin
interval = (0, 1)
train_num = 15
test_num = 10
noise = 0.5
X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise)

# 绘制曲线的数据
X_underlying = np.linspace(interval[0], interval[1], num=100)
y_underlying = sin(X_underlying)

# # 绘制图像
# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.scatter(X_train, y_train, facecolor='none', edgecolors='b', s=50, label='training data')
# plt.plot(X_underlying, y_underlying, c='g', label=r'$\sin(2\pi x)$')
# plt.legend()
# plt.savefig('多项式训练数据.jpg')
# plt.show()

# 模型训练
plt.rcParams['figure.figsize'] = (12.0, 8.0)

for i, degree in enumerate([0, 1, 3, 8]):
    model = linear(degree)
    X_train_transformed = polynomial_basic_function(X_train.reshape([-1, 1]), degree)
    X_underlying_transformed = polynomial_basic_function(X_underlying.reshape([-1, 1]), degree)
    # 训练模型得到参数
    model = optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))
    # 绘制曲线的数据
    y_underlying_pred = model(X_underlying_transformed).squeeze()
    print(model.params)

    # 绘制图像
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_train, y_train, facecolor='none', edgecolors='b', s=50, label='train data')
    plt.plot(X_underlying, y_underlying, c='g', label=r'$\sin(2\pi x)$')
    plt.plot(X_underlying, y_underlying_pred, c='r', label='predicted function')
    plt.ylim(-2, 1.5)
    plt.annotate('M={}'.format(degree), xy=(0.95, -1.4))
    plt.legend(loc='lower left', fontsize='large')

plt.savefig('不同阶数多项式训练结果.jpg')
plt.show()

# 模型评价
# 训练误差和测试误差
train_error = []
test_error = []
# sin函数与多项式回归值之间的误差
distribution_error =[]

# 遍历多项式阶数
for i in range(9):
    model = linear(i)

    X_train_transformed = polynomial_basic_function(X_train.reshape([-1 ,1]), i)
    X_test_transformed = polynomial_basic_function(X_test.reshape([-1 ,1]), i)
    X_underlying_transformed = polynomial_basic_function(X_underlying.reshape([-1 ,1]), i)

    optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))

    y_train_pred = model(X_train_transformed).squeeze()
    y_test_pred = model(X_test_transformed).squeeze()
    y_underlying_pred = model(X_underlying_transformed).squeeze()

    train_mse = mse(y_true=y_train, y_pred=y_train_pred).item()
    train_error.append(train_mse)

    test_mse = mse(y_true=y_test, y_pred=y_test_pred).item()
    test_error.append(test_mse)

print('train error:\n',train_error)
print('test error:\n',test_error)

# 可视化
plt.rcParams['figure.figsize'] = (8.0, 6.0)  # 图像尺寸
plt.plot(train_error, '-.', mfc='none', mec='r', ms=10, c='r', label='train error')
plt.plot(test_error, '-.', mfc='none', mec='g', ms=10, c='g', label='test error')
plt.legend()
plt.xlabel('degree')
plt.ylabel('MSE')
plt.savefig('多项式模型评价.jpg')
plt.show()

