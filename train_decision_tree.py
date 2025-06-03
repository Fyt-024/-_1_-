import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

def main():
    # 1. 载入数据
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
    
    X_train = train_data[['torque', 'max_power', 'km_driven']]
    y_train = train_data['selling_price']
    X_test = test_data[['torque', 'max_power', 'km_driven']]
    y_test = test_data['selling_price']

    # 2. 创建决策树回归模型，限制树深度便于可视化（例如3层）
    dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 3. 预测与评估
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Decision Tree Metrics:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R-squared:", r2)

    # 4. 可视化决策树的前几节点
    plt.figure(figsize=(12, 8))
    plot_tree(dt_model, feature_names=['torque', 'max_power', 'km_driven'], filled=True)
    plt.title("Decision Tree (max_depth=3)")
    plt.savefig("decision_tree.png")
    plt.show()
    print("决策树结构图已保存到 decision_tree.png")
    
    # 5. 保存模型
    with open("dt_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)

if __name__ == "__main__":
    main()
