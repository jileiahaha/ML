# tick 能见度
import matplotlib.pyplot as plt
import numpy as np

n = 1024    # data size
# 生成1024个呈标准正态分布的二维数据组 (平均数是0，方差为1) 作为一个数据集
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
# 每一个点的颜色值用T来表示
T = np.arctan2(Y, X)    # for color later on

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # xtick()函数来隐藏x坐标轴，y轴同理
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()