# Contours 等高线图
import matplotlib.pyplot as plt
import numpy as np

# 数据集即三维点 (x,y) 和对应的高度值，共有256个点。
# 高度值使用一个 height function f(x,y) 生成
def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

# x, y 分别是在区间 [-3,3] 中均匀分布的256个值，
# 并用meshgrid在二维平面中将每一个x和每一个y分别对应起来
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

# use plt.contourf to filling contours颜色填充
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines等高线绘制
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
# adding label添加高度数字
# inline控制是否将Label画在线里面
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()