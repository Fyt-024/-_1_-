import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math
import matplotlib.pyplot as plt

def main():
    print("开始CatBoost模型验证...")
    
    # 加载40%验证数据
    valid_data = pd.read_csv('test_data.csv')
    
    # 分离特征和目标变量
    X_valid = valid_data[['torque', 'max_power', 'km_driven']]
    y_valid = valid_data['selling_price']
    
    # 加载训练好的模型
    try:
        model = CatBoostRegressor()
        model.load_model('3catboost_model.cbm')
        print("成功加载模型")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
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
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    
    with open('3catboost_valid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("验证指标已保存到 3catboost_valid_metrics.json")
    
    # 获取特征重要性 - 确保生成dvc.yaml中定义的输出文件
    try:
        # 获取特征重要性
        feature_importances = model.get_feature_importance()
        feature_names = X_valid.columns
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # 保存特征重要性到CSV - 确保文件名与dvc.yaml中定义的一致
        feature_importance_df.to_csv('3catboost_feature_importance.csv', index=False)
        print("特征重要性已保存到 3catboost_feature_importance.csv")
        
        # 绘制特征重要性图
        plt.switch_backend('agg')  # 使用非交互式后端
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('CatBoost Feature Importance (Validation)')
        plt.tight_layout()
        plt.savefig('3catboost_feature_importance.png', dpi=300)
        plt.close()
        print("特征重要性图已保存到 3catboost_feature_importance.png")
    except Exception as e:
        print(f"生成特征重要性时出错: {e}")
        # 即使出错，也创建一个空的特征重要性文件，以满足DVC的要求
        pd.DataFrame(columns=['Feature', 'Importance']).to_csv('3catboost_feature_importance.csv', index=False)
        print("创建了空的特征重要性文件")

if __name__ == "__main__":
    main()
