# 通过引入正则项来缓解高次多项式的过拟合
import matplotlib.pyplot as plt
import paddle
from nndl.oprator import *
from nndl.dataset import *

import math


# sin函数：sin(2*pi*x)
def sin(x):
    y = paddle.sin(2 * math.pi * x)
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
X_underlying = paddle.linspace(interval[0], interval[1], num=100)
y_underlying = sin(X_underlying)


degree = 8  # 多项式阶数
reg_lambda = 0.01  # 正则项系数
X_train_transformed = polynomial_basic_function(X_train.reshape([-1, 1]), degree)  # 数据增加维度
X_test_transformed = polynomial_basic_function(X_test.reshape([-1, 1]), degree)
X_underlying_transformed = polynomial_basic_function(X_underlying.reshape([-1, 1]), degree)

model = Linear(degree)  # 定义未正则化的线性模型

optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))   # 训练未正则化的模型

y_test_pred = model(X_test_transformed).squeeze()    # 采用未正则化的训练模型，在测试集上预测标签
y_underlying_pred = model(X_underlying_transformed).squeeze()

model_reg = Linear(degree)  # 定义带正则项的线性模型

optimizer_lsm(model_reg, X_train_transformed, y_train.reshape([-1, 1]), reg_lambda=reg_lambda)  # 训练带正则项的模型

y_test_pred_reg = model_reg(X_test_transformed).squeeze()     # 采用带正则项的训练模型，在测试集上预测标签
y_underlying_pred_reg = model_reg(X_underlying_transformed).squeeze()

MSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred).item()
print('MSE:', MSE)
MSE_reg = mean_squared_error(y_true=y_test, y_pred=y_test_pred_reg).item()
print('MSE_reg:',MSE_reg)

# 可视化
plt.scatter(X_train, y_train, facecolor='none', edgecolors='r', s=50, label='train data')
plt.plot(X_underlying.numpy(), y_underlying.numpy(), c='b', label=r'$\sin(2\pi x)$')
plt.plot(X_underlying.numpy(), y_underlying_pred.numpy(), c='g', linestyle='-.', label='$deg. = 8$')
plt.plot(X_underlying.numpy(), y_underlying_pred_reg.numpy(), c='r', linestyle='-.', label='$deg. = 8, \ell_2 reg$')
plt.ylim(-1.8, 1.8)
plt.annotate('lambda={}'.format(reg_lambda), xy=(0.487, 1.53))
plt.legend()
plt.savefig('T2.3-带正则项的多项式回归.jpg')
plt.show()
