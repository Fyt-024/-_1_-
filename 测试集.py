import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 从 CSV 文件加载数据
data = pd.read_csv("filtered_data_with_median.csv")

# 2. 选取需要的列：'selling_price'、'torque'、'max_power'、'km_driven'
data = data[['selling_price', 'torque', 'max_power', 'km_driven']]

# 3. 删除包含 NaN 的行
data = data.dropna()

# 4. 定义自变量 (X) 和因变量 (y)
X = data[['torque', 'max_power', 'km_driven']]
y = data['selling_price']

# 5. 按 60% 训练集和 40% 测试集比例划分数据（这里设置 random_state=42 以保证结果可复现）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# 6. 合并测试集数据（自变量和因变量）并保存为 CSV 文件
test_data = X_test.copy()
test_data['selling_price'] = y_test
test_data.to_csv("test_data.csv", index=False)

print("测试集已成功保存到 test_data.csv 文件")
