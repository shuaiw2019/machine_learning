import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bridge.oprator import *
from Bridge.dataset import *
from Bridge.bridger import *

# 导入波士顿房价数据集
data = pd.read_csv("./data/boston_house_prices.csv", encoding='utf-8')
# 打印预览前5行数据
print(data.head())

# 查看各字段缺失值统计情况
check_out = data.isna().sum()
print(check_out)


# 箱线图查看异常值分布
def boxplot(data, fig_name):
    # 绘制每个属性的箱线图
    data_col = list(data.columns)

    # 连续画几个图片
    plt.figure(figsize=(5, 5), dpi=300)
    # 子图调整
    plt.subplots_adjust(wspace=0.6)
    # 每个特征画一个箱线图
    for i, col_name in enumerate(data_col):
        plt.subplot(3, 5, i + 1)
        # 画箱线图
        plt.boxplot(data[col_name],
                    showmeans=True,
                    meanprops={"markersize": 1, "marker": "D", "markeredgecolor": "#C54680"},  # 均值的属性
                    medianprops={"color": "#946279"},  # 中位数线的属性
                    whiskerprops={"color": "#8E004D", "linewidth": 0.4, 'linestyle': "--"},
                    flierprops={"markersize": 0.4},
                    )
        # 图名
        plt.title(col_name, fontdict={"size": 5}, pad=2)
        # y方向刻度
        plt.yticks(fontsize=4, rotation=90)
        plt.tick_params(pad=0.5)
        # x方向刻度
        plt.xticks([])
    plt.savefig(fig_name)
    plt.show()


# boxplot(data, '波士顿房价数据_处理前.jpg')

# 四分位处理异常值
num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()

for feature in num_features:
    if feature == 'CHAS':
        continue

    Q1 = data[feature].quantile(q=0.25)  # 下四分位
    Q3 = data[feature].quantile(q=0.75)  # 上四分位

    IQR = Q3 - Q1
    top = Q3 + 1.5 * IQR  # 最大估计值
    bot = Q1 - 1.5 * IQR  # 最小估计值
    values = data[feature].values
    values[values > top] = top  # 临界值取代噪声
    values[values < bot] = bot  # 临界值取代噪声
    data[feature] = values.astype(data[feature].dtype)

# 再次查看箱线图
# boxplot(data, '波士顿房价数据_处理后.jpg')


# 数据集划分
np.random.seed(10)
# 划分训练集和测试集
def train_test_split(X, y, train_percent=0.8):
    n = len(X)
    shuffled_indices = np.random.permutation(n)
    train_set_size = int(n * train_percent)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]

    X = X.values
    y = y.values

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


X = data.drop(['MEDV'], axis=1)   # 删除MEDV列
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y)

y_train = y_train.reshape([-1,1])
y_test = y_test.reshape([-1,1])


# 特征工程
# X_train = np.array([X_train], dtype='float32')
# y_train = np.array([y_train], dtype='float32')
# X_test = np.array([X_test], dtype='float32')
# y_test = np.array([y_test], dtype='float32')

X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

X_train = (X_train - X_min)/(X_max - X_min)
X_test = (X_test - X_min)/(X_max - X_min)

train_dataset = (X_train, y_train)
test_dataset = (X_test, y_test)


# 构建模型
input_size = 12
model = linear(input_size)


# 模型训练
bridger = bridger(model=model, optimizer=optimizer_lsm, loss_fn=None, metric=mse)   # 实例化
# 模型保存文件夹
saved_dir = './model'
bridger.train(dataset=train_dataset, reg_lambda=0, model_dir=saved_dir)

columns_list = data.columns.to_list()
weights = bridger.model.params['w'].tolist()
b = bridger.model.params['b'].item()

for i in range(len(weights)):
    print(columns_list[i],"weight:",weights[i])

print("b:",b)


# 加载模型权重
bridger.load_model(saved_dir)

MSE = bridger.evaluate(test_dataset)
print('MSE:', MSE.item())


bridger.load_model(saved_dir)
pred = bridger.predict(X_test[:1])
print("真实房价：",y_test[:1].item())
print("预测的房价：",pred.item())