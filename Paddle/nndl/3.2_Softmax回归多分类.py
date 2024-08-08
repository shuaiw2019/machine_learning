from cooker.dataset import *
from cooker.oprator import *
from cooker.optimizer import *
from cooker.runner import *
from nndl.visul import *

paddle.seed(62)
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


input_dim = 2     # 特征维度
output_dim = 3    # 类别数
lr = 0.1          # 学习率
# 实例化模型
model =ModelSR(input_dim=input_dim, output_dim=output_dim)
# 指定优化器
optimizer = SimpleBatchGD(init_lr=lr, model=model)
# 指定损失函数
loss_fn = MultiCrossEntroLoss()
# 指定评价指标
accuracy = accuracy
# 实例化bridger类
runner = Runner(model=model, optimizer=optimizer, loss_fn=loss_fn, metric=accuracy)
# 模型训练
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=500, log_epochs=50, eval_epochs=1, save_path='3.2_best.model')
# 可视化训练结果
# plot_class(runner, fig_name='T3.2-多分类训练结果.jpg')

# 模型评价
score, loss = runner.evaluate([X_test, y_test])
print('[test] score / loss:{:.4f}/{:.4f}'.format(score, loss))

# 可视化分类边界
# 均匀生成40,000个数据点
x1, x2 = paddle.meshgrid(paddle.linspace(-3.5, 2, 200), paddle.linspace(-4.5, 3.5, 200))
x = paddle.stack([paddle.flatten(x1), paddle.flatten(x2)], axis=1)
# 预测对应类别
y = runner.predict(x)
y = paddle.argmax(y, axis=1)
# 绘制类别区域
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(x[:,0].tolist(), x[:,1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)

paddle.seed(62)
n_samples = 1000
X, y = make_multiclass(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)
plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())
plt.show()
