import json
import os
import pandas as pd

# 定义要读取的文件模式
file_patterns = {
    "线性回归": {
        "train": "1linear_reg_train_metrics.json",
        "valid": "1linear_reg_valid_metrics.json",
        "full": "1linear_reg_full_metrics.json"
    },
    "决策树": {
        "train": "2decision_tree_train_metrics.json",
        "valid": "2decision_tree_valid_metrics.json",
        "full": "2decision_tree_full_metrics.json"
    },
    "CatBoost": {
        "train": "3catboost_metrics.json",
        "valid": "3catboost_valid_metrics.json",
        "full": "3catboost_full_metrics.json"
    },
    "XGBoost": {
        "train": "4xgboost_metrics.json",
        "valid": "4xgboost_valid_metrics.json",
        "full": "4xgboost_full_metrics.json"
    },
    "神经网络": {
        "train": "5nn_metrics.json",
        "valid": "5nn_valid_metrics.json",
        "full": "5nn_full_metrics.json"
    }
}

# 创建结果表格
results = []

for model_name, files in file_patterns.items():
    row = {"模型": model_name}
    
    # 读取训练集指标
    try:
        if os.path.exists(files["train"]):
            with open(files["train"], "r") as f:
                train_metrics = json.load(f)
                # 尝试不同的键名，因为不同模型可能使用不同的键
                rmse = train_metrics.get("rmse", train_metrics.get("RMSE", "N/A"))
                r2 = train_metrics.get("r2", train_metrics.get("R2", train_metrics.get("r^2", "N/A")))
                row["训练集RMSE"] = rmse
                row["训练集R²"] = r2
        else:
            row["训练集RMSE"] = "文件不存在"
            row["训练集R²"] = "文件不存在"
    except Exception as e:
        row["训练集RMSE"] = f"错误: {str(e)}"
        row["训练集R²"] = f"错误: {str(e)}"
    
    # 读取验证集指标
    try:
        if os.path.exists(files["valid"]):
            with open(files["valid"], "r") as f:
                valid_metrics = json.load(f)
                rmse = valid_metrics.get("rmse", valid_metrics.get("RMSE", "N/A"))
                r2 = valid_metrics.get("r2", valid_metrics.get("R2", valid_metrics.get("r^2", "N/A")))
                row["验证集RMSE"] = rmse
                row["验证集R²"] = r2
        else:
            row["验证集RMSE"] = "文件不存在"
            row["验证集R²"] = "文件不存在"
    except Exception as e:
        row["验证集RMSE"] = f"错误: {str(e)}"
        row["验证集R²"] = f"错误: {str(e)}"
    
    # 读取完整数据集指标
    try:
        if os.path.exists(files["full"]):
            with open(files["full"], "r") as f:
                full_metrics = json.load(f)
                rmse = full_metrics.get("rmse", full_metrics.get("RMSE", "N/A"))
                r2 = full_metrics.get("r2", full_metrics.get("R2", full_metrics.get("r^2", "N/A")))
                row["完整数据集RMSE"] = rmse
                row["完整数据集R²"] = r2
        else:
            row["完整数据集RMSE"] = "文件不存在"
            row["完整数据集R²"] = "文件不存在"
    except Exception as e:
        row["完整数据集RMSE"] = f"错误: {str(e)}"
        row["完整数据集R²"] = f"错误: {str(e)}"
    
    results.append(row)

# 创建DataFrame
df = pd.DataFrame(results)

# 保存为CSV
df.to_csv("model_comparison.csv", index=False, encoding='utf-8-sig')

# 打印表格
print("\n模型性能比较表：")
print(df.to_string(index=False))

print("\n表格已保存为 model_comparison.csv")
