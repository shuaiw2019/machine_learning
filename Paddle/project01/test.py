import numpy as np
import matplotlib.pyplot as plt
import paddle


def logistic(x):
    return 1 / (1 + paddle.exp(-x))


def swish(x, beta=1):
    return x * logistic(beta * x)


def plt_fn(*functions_with_args):
    x = paddle.linspace(-10, 10, 10000)
    plt.figure()  # 创建新图像
    for i, (func, args) in enumerate(functions_with_args):
        # 使用不同的颜色和线型
        color = f"C{i}"  # Matplotlib默认颜色循环
        linestyle = '-' if i % 2 == 0 else '--'  # 偶数索引使用实线，奇数索引使用虚线
        y = func(x, *args)  # 传入参数
        plt.plot(x.tolist(), y.tolist(), color=color, linestyle=linestyle, label=f'{func.__name__}({args})')

    ax = plt.gca()  # 获取当前坐标轴对象
    ax.spines['top'].set_color('none')  # 取消上侧坐标轴
    ax.spines['right'].set_color('none')  # 取消右侧坐标轴
    ax.xaxis.set_ticks_position('bottom')  # 设置默认的x坐标轴
    ax.yaxis.set_ticks_position('left')  # 设置默认的y坐标轴
    ax.spines['left'].set_position(('data', 0))  # 设置y轴起点为0
    ax.spines['bottom'].set_position(('data', 0))  # 设置x轴起点为0

    plt.legend(loc=0, fontsize='medium')  # 图例
    plt.savefig(f'T4.1-{"、".join(func.__name__ for func, _ in functions_with_args)}函数.jpg')  # 保存图片
    plt.show()  # 显示图像


if __name__ == '__main__':
    betas = [0, 0.5, 1, 100]
    functions_with_args = [(swish, (beta,)) for beta in betas]  # 列表推导式
    plt_fn(*functions_with_args)