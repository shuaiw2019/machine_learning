from nndl import oprator
from nndl import dataset
import numpy as np
import math
import paddle


x1 = paddle.linspace(-10, 10, 10000)
x2 = np.linspace(-10, 10, 10)

print(paddle.exp(x1))
print(np.exp(x2))
print(math.exp(2))