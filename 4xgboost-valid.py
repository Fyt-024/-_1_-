import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math
# 在导入matplotlib之前设置非交互式后端
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要GUI
import matplotlib.pyplot as plt

def main():
    print("开始XGBoost模型验证...")
    
    # 加载40%验证数据
    valid_data = pd.read_csv('test_data.csv')
    
    # 分离特征和目标变量
    X_valid = valid_data[['torque', 'max_power', 'km_driven']]
    y_valid = valid_data['selling_price']
    
    # 创建DMatrix对象
    dvalid = xgb.DMatrix(X_valid)
    
    # 加载训练好的模型
    model = xgb.Booster()
    model.load_model('4xgb_model.json')
    
    # 在验证集上评估模型
    y_pred = model.predict(dvalid)
    
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
    
    with open('4xgboost_valid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("验证指标已保存到 4xgboost_valid_metrics.json")
    
    # 获取特征重要性分数
    importance_scores = model.get_score(importance_type='weight')
    
    # 创建特征重要性DataFrame
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # 如果特征名称是f0, f1, f2格式，转换为实际特征名称
    if all(f.startswith('f') for f in features):
        feature_names = X_valid.columns
        features = [feature_names[int(f[1:])] for f in features]
    
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': scores
    })
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # 保存特征重要性到CSV
    feature_importance_df.to_csv('4xgboost_feature_importance.csv', index=False)
    print("特征重要性已保存到 4xgboost_feature_importance.csv")
    
    # 创建特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('重要性分数')
    plt.ylabel('特征')
    plt.title('XGBoost特征重要性 (验证集)')
    plt.tight_layout()
    plt.savefig('4xgboost_feature_importance.png')
    print("特征重要性图已保存到 4xgboost_feature_importance.png")

if __name__ == "__main__":
    main()
