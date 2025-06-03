import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import math

def main():
    print("开始决策树模型验证...")
    
    # 加载40%验证数据
    valid_data = pd.read_csv('test_data.csv')
    
    # 分离特征和目标变量
    X_valid = valid_data[['torque', 'max_power', 'km_driven']]
    y_valid = valid_data['selling_price']
    
    # 加载训练好的模型
    with open('2dt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_valid)
    
    # 计算评估指标
    mse = mean_squared_error(y_valid, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    print(f"验证集评估结果:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    
    # 保存评估指标
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    with open('2decision_tree_valid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("验证指标已保存到 decision_tree_valid_metrics.json")

if __name__ == "__main__":
    main()
