import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. 从 CSV 文件中加载数据
data = pd.read_csv("filtered_data_with_median.csv")

# 2. 选取需要的列：'selling_price', 'torque', 'max_power', 'km_driven'
data = data[['selling_price', 'torque', 'max_power', 'km_driven']]

# 3. 删除包含 NaN 值的行
data = data.dropna()

# 4. 定义特征（X）和目标变量（y）
X = data[['torque', 'max_power', 'km_driven']]
y = data['selling_price']

# 5. 将数据按 60%（训练集）和 40%（测试集）划分（这里设置 random_state=42 以保证结果可复现）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# 6. 使用训练集数据构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 输出模型参数及训练集得分以供参考
print("模型系数 Model coefficients:", model.coef_)
print("截距 Intercept:", model.intercept_)
print("训练集得分 Training set score:", model.score(X_train, y_train))

# 8. 重新组合训练集数据（包含特征和目标变量）并保存为 CSV 文件
train_data = X_train.copy()
train_data['selling_price'] = y_train
train_data.to_csv("train_data.csv", index=False)


