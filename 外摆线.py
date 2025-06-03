import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========= 参数设置 =========
R = 1.2      # 大圆半径
r = 0.2      # 小圆半径（调整为0.2以满足6个花瓣： (1+0.2)/0.2 = 6）
d = 0.5        # 描迹点到小圆圆心的距离

# ========= 生成外摆线轨迹 =========
theta = np.linspace(0, 2*np.pi, 1000)
x = (R + r) * np.cos(theta) - d * np.cos(((R + r)/r) * theta)
y = (R + r) * np.sin(theta) - d * np.sin(((R + r)/r) * theta)

# ========= 创建画布 =========
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(- (2*R + 2*r), (2*R + 2*r))
ax.set_ylim(- (2*R + 2*r), (2*R + 2*r))
ax.set_title("6瓣外摆线动画")

# 绘制轨迹线和运动点
(line,) = ax.plot([], [], 'r-', lw=2)
(point,) = ax.plot([], [], 'bo', ms=6)

# ========= 动画初始化函数 =========
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# ========= 动画更新函数 =========
def update(frame):
    frame = int(frame)  # 确保 frame 为整数
    line.set_data(x[:frame], y[:frame])
    # 将单个点的坐标包装为列表
    point.set_data([x[frame]], [y[frame]])
    return line, point

# ========= 创建动画 =========
ani = FuncAnimation(fig, update, frames=len(theta),
                    init_func=init, interval=10, blit=True)

plt.show()