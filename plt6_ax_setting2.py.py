# 设置坐标轴2
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()
plt.plot(x,y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')

#设置x轴坐标点-1到2范围，等距设置5个点
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)

plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])


# 使用plt.gca获取当前坐标轴信息.
# 使用.spines设置边框：右侧边框；
# 使用.set_color设置边框颜色：默认白色；
# 使用.spines设置边框：上边框；
# 使用.set_color设置边框颜色：默认白色；
ax = plt.gca()
ax.spines['right'].set_color('red')
ax.spines['top'].set_color('red')


# 调整坐标轴

# 使用.xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：
# bottom.（所有位置：top，bottom，both，default，none）
ax.xaxis.set_ticks_position('bottom')

# 使用.spines设置边框：x轴；
# 使用.set_position设置边框位置：y=0的位置；
# （位置所有属性：outward，axes，data）
ax.spines['bottom'].set_position(('data', 0))

# 使用.yaxis.set_ticks_position设置y坐标刻度数字或名称的位置：
# left.（所有位置：left，right，both，default，none）
ax.yaxis.set_ticks_position('left')

ax.spines['left'].set_position(('data',0))

plt.show()