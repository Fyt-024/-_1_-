import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def main():
    # 1. 载入数据
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
    X_train = train_data[['torque', 'max_power', 'km_driven']]
    y_train = train_data['selling_price']
    X_test = test_data[['torque', 'max_power', 'km_driven']]
    y_test = test_data['selling_price']

    # 2. 构建 CatBoost 回归模型
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. 预测与评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("CatBoost Metrics:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R-squared:", r2)
    
    # 4. 输出特征重要性
    importance = model.get_feature_importance()
    features = ['torque', 'max_power', 'km_driven']
    feature_importance = dict(zip(features, importance))
    print("Feature Importance:", feature_importance)
    
    # 保存特征重要性到 CSV 文件
    pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance']).to_csv("catboost_feature_importance.csv", index=False)
    
    # 5. 保存指标
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    with open("catboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 6. 保存模型
    model.save_model("catboost_model.cbm")
    
if __name__ == "__main__":
    main()
