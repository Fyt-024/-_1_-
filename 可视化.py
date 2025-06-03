import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectories():
    # 检查文件是否存在
    if not os.path.exists('trajectory_data.csv'):
        print("Error: trajectory_data.csv not found!")
        return
    
    if not os.path.exists('expected_trajectory.csv'):
        print("Error: expected_trajectory.csv not found!")
        return
    
    # 读取实际轨迹数据
    try:
        actual_data = pd.read_csv('trajectory_data.csv')
        print(f"Loaded actual trajectory data with {len(actual_data)} points")
    except Exception as e:
        print(f"Error loading actual trajectory data: {e}")
        return
    
    # 读取预期轨迹数据
    try:
        expected_data = pd.read_csv('expected_trajectory.csv')
        print(f"Loaded expected trajectory data with {len(expected_data)} points")
    except Exception as e:
        print(f"Error loading expected trajectory data: {e}")
        return
    
    # 创建轨迹对比图
    plt.figure(figsize=(12, 10))
    
    # 绘制预期轨迹
    plt.plot(expected_data['X'], expected_data['Y'], 'b-', linewidth=2, label='Expected Trajectory')
    
    # 绘制实际轨迹
    plt.plot(actual_data['X_Actual'], actual_data['Y_Actual'], 'r-', linewidth=2, label='Actual Trajectory')
    
    # 标记起点
    plt.plot(actual_data['X_Actual'].iloc[0], actual_data['Y_Actual'].iloc[0], 'go', markersize=10, label='Start Point')
    
    # 添加网格和图例
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('Robot Trajectory Comparison', fontsize=16)
    plt.xlabel('X Position (m)', fontsize=14)
    plt.ylabel('Y Position (m)', fontsize=14)
    
    # 确保X和Y轴比例相同
    plt.axis('equal')
    
    # 计算误差统计
    if 'Error_Distance' in actual_data.columns:
        error_distance = actual_data['Error_Distance']
    else:
        error_x = actual_data['X_Expected'] - actual_data['X_Actual']
        error_y = actual_data['Y_Expected'] - actual_data['Y_Actual']
        error_distance = np.sqrt(error_x**2 + error_y**2)
    
    avg_error = error_distance.mean()
    max_error = error_distance.max()
    
    # 添加误差信息
    plt.figtext(0.5, 0.01, f'Average Error: {avg_error:.4f} m, Maximum Error: {max_error:.4f} m', 
                ha='center', fontsize=12, bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
    
    # 保存图像
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved trajectory comparison to trajectory_comparison.png")
    
    # 创建误差随时间变化图
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data['Time']/1000, error_distance, 'g-', linewidth=2)
    plt.grid(True)
    plt.title('Tracking Error Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Error Distance (m)', fontsize=14)
    plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average Error: {avg_error:.4f} m')
    plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average Error: {avg_error:.4f} m')
    plt.legend(fontsize=12)
    
    # 保存误差图
    plt.savefig('error_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved error plot to error_over_time.png")
    
    # 创建速度图
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data['Time']/1000, np.sqrt(actual_data['VX']**2 + actual_data['VY']**2), 'b-', linewidth=2)
    plt.grid(True)
    plt.title('Robot Speed Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Speed (m/s)', fontsize=14)
    
    # 保存速度图
    plt.savefig('speed_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved speed plot to speed_over_time.png")
    
    # 创建热力图显示误差分布
    plt.figure(figsize=(12, 10))
    
    # 创建散点图，颜色表示误差大小
    scatter = plt.scatter(actual_data['X_Actual'], actual_data['Y_Actual'], 
                         c=error_distance, cmap='jet', s=30, alpha=0.7)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Distance (m)', fontsize=12)
    
    # 添加预期轨迹作为参考
    plt.plot(expected_data['X'], expected_data['Y'], 'k--', linewidth=1, alpha=0.5, label='Expected Path')
    
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('Error Distribution Along Trajectory', fontsize=16)
    plt.xlabel('X Position (m)', fontsize=14)
    plt.ylabel('Y Position (m)', fontsize=14)
    plt.axis('equal')
    
    # 保存热力图
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved error distribution plot to error_distribution.png")
    
    # 显示所有图形
    plt.show()

if __name__ == "__main__":
    plot_trajectories()