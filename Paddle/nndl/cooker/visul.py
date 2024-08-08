from matplotlib import pyplot as plt


# 可视化观察训练集与验证集的指标变化情况
def plot_class(bridger, fig_name):
    # 图像尺寸
    plt.figure(figsize=(10, 5))
    # 子图拆分,损失变化情况
    plt.subplot(1,2,1)
    # 利用训练得分计算回合总数
    epochs = [i for i in range(len(bridger.train_scores))]
    # 绘制训练损失变化曲线
    plt.plot(epochs, bridger.train_loss, color='#8E004D', label="Train loss")
    # 绘制评价损失变化曲线
    plt.plot(epochs, bridger.dev_loss, color='#E20079', linestyle='--', label="Dev loss")
    # 绘制坐标轴和图例
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc='upper right')
    # 子图拆分,得分变化情况
    plt.subplot(1,2,2)
    # 绘制训练准确率变化曲线
    plt.plot(epochs, bridger.train_scores, color='#8E004D', label="Train accuracy")
    # 绘制评价准确率变化曲线
    plt.plot(epochs, bridger.dev_scores, color='#E20079', linestyle='--', label="Dev accuracy")
    # 绘制坐标轴和图例
    plt.ylabel("score")
    plt.xlabel("epoch")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
