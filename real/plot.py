import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据文件名为data.txt
# filename = './action_values_cpp.txt'
filename = './obs_values_cpp.txt'

# 读取数据
with open(filename, 'r') as file:
    data = file.readlines()

# 将数据转换为浮点数，并将其转换为numpy数组
data = [list(map(float, row.split())) for row in data]

# 获取时间点数量（行数）
time_points = len(data)

# 创建时间轴（单位是秒）
time_axis = np.linspace(0, time_points - 1, time_points)

# 绘制每一列的数据
plt.figure(figsize=(10, 6))

for i in range(12):
    plt.plot(time_axis, [row[i] for row in data], label=f'Joint {i+1}')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Joint Angles Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Joint Angle')

# 显示图表
plt.show()