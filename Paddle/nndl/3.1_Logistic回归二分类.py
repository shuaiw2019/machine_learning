from nndl.dataset import *
from nndl.oprator import *
from cooker.runner import *
from cooker.optimizer import *
import matplotlib.pyplot as plt


# 采样1000个样本
n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.5)
# # 可视化生产的数据集，不同颜色代表不同类别
# plt.figure(figsize=(5,5))
# plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
# plt.xlim(-3,4)
# plt.ylim(-3,4)
# plt.savefig('T3.1-Logistic回归数据.jpg')
# plt.show()

# 将1000条样本数据拆分成训练集、验证集和测试集，其中训练集640条、验证集160条、测试集200条
num_train = 640
num_dev = 160
num_test = 200

# 转换前：y_train.shape=[640]
X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

# 转换前：y_train.shape=[640, 1]
y_train = y_train.reshape([-1, 1])
y_dev = y_dev.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])

# print(y_train[:5])

paddle.seed(102)

# 模型训练
# 超参数
input_dim = 2    # 特征维度
lr = 0.1         # 学习率
# 实例化模型
model = ModelLR(input_dim=input_dim)
# 指定优化器
optimizer = SimpleBatchGD(init_lr=lr, model=model)
# 指定损失函数
loss_fn = BinaryCrossEntropyLoss()
# 指定评价方式
metric = accuracy
# 实例化bridger类，并传入训练配置
runner = Runner(model=model, optimizer=optimizer, loss_fn=loss_fn, metric=metric)
runner.train([X_train,y_train], [X_dev, y_dev], num_epochs=500, log_epochs=50, save_path='best_model.pdparams')

# 训练结果可视化
# plot(runner, fig_name='T3.1-二分类训练结果.jpg')

# 模型评价
score, loss = runner.evaluate([X_test, y_test])
print('[test] score/loss: {:.4f}/{:.4f}'.format(score, loss))


# 可视化分类边界
def decision_boundary(w, b, x1):
    w1, w2 = w
    x2 = (- w1 * x1 - b) / w2
    return x2


plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())

w = model.params['w']
b = model.params['b']
x1 = paddle.linspace(-2,3, 1000)
x2 = decision_boundary(w, b, x1)
plt.plot(x1.tolist(), x2.tolist(), color='r')
plt.show()
