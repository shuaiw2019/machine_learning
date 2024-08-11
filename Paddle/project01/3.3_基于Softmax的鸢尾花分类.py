from sklearn.datasets import load_iris
import pandas
import paddle
import numpy as np
from nndl import dataset, oprator, optimizer, metric, runner, visul

# 读取鸢尾花数据
iris_features = np.array(load_iris().data, dtype=np.float32)
iris_labels = np.array(load_iris().target, dtype=np.int32)
# 打印特征、标签数据形状
print(iris_features.shape)
print(iris_labels.shape)
# 缺失数据检查
print(pandas.isna(iris_features).sum())
print(pandas.isna(iris_labels).sum())
# # 异常数据分析
# title = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# dataset.boxplot_2(iris_features, size=(5, 5), title=title, figname='T3.3-鸢尾花异常数据检查.jpg')

# 数据处理
X = paddle.to_tensor(iris_features)
y = paddle.to_tensor(iris_labels)
# 归一化
X_max = paddle.max(X)
X_min = paddle.min(X)
X = (X - X_min) / (X_max - X_min)
# 打乱数据顺序
paddle.seed(180)
idx = paddle.randperm(X.shape[0])
X = X[idx]
y = y[idx]
# 拆分数据集
train_num = 120
dev_num = 15
test_num = 15
X_train, y_train = X[:train_num], y[:train_num]
X_dev, y_dev = X[train_num:train_num + dev_num], y[train_num:train_num + dev_num]
X_test, y_test = X[train_num + dev_num:], y[train_num + dev_num:]
# print(X_train.shape)
# print(X_dev.shape)
# print(X_test.shape)

# 模型构建
input_dim =4
output_dim = 3
model = oprator.ModelSR(input_dim, output_dim)

# 模型训练
# -学习率
lr = 0.05
# -优化器
optimizer = optimizer.SimpleBatchGD(init_lr=lr, model=model)
# -损失函数
loss_fn = oprator.MultiCrossEntroLoss()
# -评价指标
metric = metric.accuracy
# -实例化runner类
runner = runner.Runner(model=model, optimizer=optimizer,loss_fn=loss_fn,metric=metric)
# -启动训练
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=400, log_epochs=10,
             save_path='./model/3.3-best_model')
# -训练结果可视化
visul.plot_train_out(runner, fig_name='T3.3-基于Softmax的鸢尾花分类.jpg')

# 模型评价
# -加载训练模型
runner.load_model('./model/3.3-best_model')
# -评估
score, loss = runner.evaluate([X_test, y_test])
print(f'模型在测试集上的精度为：{score},损失值为：{loss}')

# 模型预测
# -计算测试集所有数据
logits = runner.predict(X_test)
# -获取第1条数据的预测类别
pred_1 = paddle.argmax(logits[0]).numpy()
# -获取第1条数据的真实类别
label_1 = y_test[0].numpy()
print(f'第1条数据的预测类别为：{pred_1},实际类别为：{label_1}')
