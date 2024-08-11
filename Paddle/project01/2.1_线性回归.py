import paddle
from matplotlib import pyplot as plt
from nndl.dataset import *
from nndl.metric import *
from nndl.oprator import *
from nndl.optimizer import *


# 线性函数
def linear_func(x, w = 1.2, b = 0.5):
    y = x * w + b
    return y


func = linear_func
interval = (-10, 10)
train_num = 100  # 训练样本数目
test_num = 50  # 测试样本数目
noise = 2
X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False)

# 生成数据用于绘制直线
# paddle.linspace返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num
X_underlying = paddle.linspace(interval[0], interval[1], train_num)
y_underlying = linear_func(X_underlying)

# 绘制数据
plt.scatter(X_train, y_train, marker='*', facecolor="none", edgecolor='#e4007f', s=50, label="train data")
plt.scatter(X_test, y_test, facecolor="none", edgecolor='#f19ec2', s=50, label="test data")
plt.plot(X_underlying.numpy(), y_underlying.numpy(), c='#000000', label=r"underlying distribution")
plt.legend()
plt.savefig('线性回归训练数据')
plt.show()

# 模型训练，100份数据
input_size = 1
reg_lambda = 0
model = Linear(input_size)

model = optimizer_lsm(model, X_train.reshape([-1, 1]), y_train.reshape([-1, 1]), reg_lambda=reg_lambda)
print('w_pred:', model.params['w'].item(), 'b_pred:', model.params['b'].item())

y_train_pred = model(X_train.reshape([-1, 1])).squeeze()
train_error = mean_squared_error(y_true=y_train,y_pred=y_train_pred).item()
print('train error:', train_error)

# 模型评价，50份数据
y_test_prd = model(X_test.reshape([-1, 1])).squeeze()
test_error = mean_squared_error(y_true=y_test, y_pred=y_test_prd).item()
print('test error:', test_error)