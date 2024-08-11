import matplotlib.pyplot as plt
import numpy as np

# 创建一些随机数据
data = np.random.randn(100)

# 创建箱形图
plt.boxplot(data,
            vert=True,  # 竖直方向绘制箱形图
            patch_artist=True,  # 填充箱体
            notch=False,  # 不使用凹口
            showmeans=False,  # 不显示均值
            meanline=False,  # 不使用线来表示均值
            showfliers=True,  # 显示异常值
            medianprops=dict(color="red"),  # 设置中位数线的颜色
            whiskerprops=dict(color="green", linewidth=2),  # 设置须的颜色和宽度
            capprops=dict(color="blue"),  # 设置端点的颜色
            boxprops=dict(facecolor="lightblue"),  # 设置箱体的颜色
            flierprops=dict(markerfacecolor="red", markersize=6))  # 设置异常值的颜色和大小

# 显示图形
plt.show()