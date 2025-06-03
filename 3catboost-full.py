import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math

def main():
    print("开始CatBoost模型全数据集评估...")
    
    # 加载全部数据
    full_data = pd.read_csv('preprocessed_data.csv')
    
    # 分离特征和目标变量
    X_full = full_data[['torque', 'max_power', 'km_driven']]
    y_full = full_data['selling_price']
    
    # 加载训练好的模型
    try:
        model = CatBoostRegressor()
        model.load_model('3catboost_model.cbm')  # 修改这里，使用正确的文件名
        print("成功加载模型")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 在全数据集上评估模型
    y_pred = model.predict(X_full)
    
    # 计算评估指标
    mse = mean_squared_error(y_full, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_full, y_pred)
    r2 = r2_score(y_full, y_pred)
    
    print(f"全数据集评估结果:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    
    # 保存评估指标
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    
    with open('3catboost_full_metrics.json', 'w') as f:  # 确保文件名与dvc.yaml中定义的一致
        json.dump(metrics, f, indent=4)
    print("全数据集评估指标已保存到 3catboost_full_metrics.json")

if __name__ == "__main__":
    main()
