import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
import os

def main():
    print("开始在验证集上评估模型...")
    
    # 检查验证数据是否存在，如果不存在，则创建
    if not os.path.exists('validation_data.csv'):
        print("验证数据不存在，正在从原始数据创建验证集...")
        # 加载原始数据
        try:
            full_data = pd.read_csv('data.csv')
            # 假设我们已经有了训练数据，我们将剩余的30%作为验证集
            train_data = pd.read_csv('train_data.csv')
            
            # 获取训练集中的所有ID
            train_ids = set(train_data.index)
            
            # 选择不在训练集中的数据作为验证集
            validation_data = full_data[~full_data.index.isin(train_ids)]
            
            # 保存验证数据
            validation_data.to_csv('validation_data.csv', index=False)
            print(f"验证集已创建，包含 {len(validation_data)} 条记录")
        except Exception as e:
            print(f"创建验证集时出错: {e}")
            # 如果没有原始数据，则从训练数据中分割
            try:
                train_data = pd.read_csv('train_data.csv')
                # 随机选择20%的训练数据作为验证集
                validation_data = train_data.sample(frac=0.2, random_state=42)
                train_data = train_data.drop(validation_data.index)
                
                # 保存更新后的训练数据和验证数据
                train_data.to_csv('train_data.csv', index=False)
                validation_data.to_csv('validation_data.csv', index=False)
                print(f"从训练数据中分割出验证集，包含 {len(validation_data)} 条记录")
            except Exception as e:
                print(f"从训练数据分割验证集时出错: {e}")
                return
    
    # 加载验证数据
    validation_data = pd.read_csv('validation_data.csv')
    print(f"加载验证数据，包含 {len(validation_data)} 条记录")
    
    # 分离特征和目标变量
    X_val = validation_data[['torque', 'max_power', 'km_driven']]
    y_val = validation_data['selling_price']
    
    # 评估XGBoost模型
    try:
        # 加载XGBoost模型
        xgb_model = xgb.Booster()
        xgb_model.load_model('4xgb_model.json')
        
        # 转换数据为DMatrix格式
        dval = xgb.DMatrix(X_val)
        
        # 预测
        y_pred_xgb = xgb_model.predict(dval)
        
        # 计算评估指标
        mse_xgb = mean_squared_error(y_val, y_pred_xgb)
        rmse_xgb = math.sqrt(mse_xgb)
        mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
        r2_xgb = r2_score(y_val, y_pred_xgb)
        
        print("\nXGBoost模型在验证集上的评估结果:")
        print(f"MSE: {mse_xgb}")
        print(f"RMSE: {rmse_xgb}")
        print(f"MAE: {mae_xgb}")
        print(f"R2: {r2_xgb}")
        
        # 保存评估指标
        metrics_xgb = {
            "validation_mse": float(mse_xgb),
            "validation_rmse": float(rmse_xgb),
            "validation_mae": float(mae_xgb),
            "validation_r2": float(r2_xgb)
        }
        
        with open('4xgboost_validation_metrics.json', 'w') as f:
            json.dump(metrics_xgb, f, indent=4)
        print("XGBoost验证指标已保存到 4xgboost_validation_metrics.json")
        
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred_xgb, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('XGBoost模型: 预测值 vs 实际值')
        plt.savefig('4xgboost_validation_plot.png')
        plt.close()
        print("XGBoost验证对比图已保存到 4xgboost_validation_plot.png")
    except Exception as e:
        print(f"评估XGBoost模型时出错: {e}")
    
    # 评估CatBoost模型
    try:
        from catboost import CatBoostRegressor
        
        # 加载CatBoost模型
        cb_model = CatBoostRegressor()
        cb_model.load_model('3catboost_model.cbm')
        
        # 预测
        y_pred_cb = cb_model.predict(X_val)
        
        # 计算评估指标
        mse_cb = mean_squared_error(y_val, y_pred_cb)
        rmse_cb = math.sqrt(mse_cb)
        mae_cb = mean_absolute_error(y_val, y_pred_cb)
        r2_cb = r2_score(y_val, y_pred_cb)
        
        print("\nCatBoost模型在验证集上的评估结果:")
        print(f"MSE: {mse_cb}")
        print(f"RMSE: {rmse_cb}")
        print(f"MAE: {mae_cb}")
        print(f"R2: {r2_cb}")
        
        # 保存评估指标
        metrics_cb = {
            "validation_mse": float(mse_cb),
            "validation_rmse": float(rmse_cb),
            "validation_mae": float(mae_cb),
            "validation_r2": float(r2_cb)
        }
        
        with open('3catboost_validation_metrics.json', 'w') as f:
            json.dump(metrics_cb, f, indent=4)
        print("CatBoost验证指标已保存到 3catboost_validation_metrics.json")
        
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred_cb, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('CatBoost模型: 预测值 vs 实际值')
        plt.savefig('3catboost_validation_plot.png')
        plt.close()
        print("CatBoost验证对比图已保存到 3catboost_validation_plot.png")
    except Exception as e:
        print(f"评估CatBoost模型时出错: {e}")
    
    # 评估决策树模型
    try:
        # 加载决策树模型
        with open('dt_model.pkl', 'rb') as f:
            dt_model = pickle.load(f)
        
        # 预测
        y_pred_dt = dt_model.predict(X_val)
        
        # 计算评估指标
        mse_dt = mean_squared_error(y_val, y_pred_dt)
        rmse_dt = math.sqrt(mse_dt)
        mae_dt = mean_absolute_error(y_val, y_pred_dt)
        r2_dt = r2_score(y_val, y_pred_dt)
        
        print("\n决策树模型在验证集上的评估结果:")
        print(f"MSE: {mse_dt}")
        print(f"RMSE: {rmse_dt}")
        print(f"MAE: {mae_dt}")
        print(f"R2: {r2_dt}")
        
        # 保存评估指标
        metrics_dt = {
            "validation_mse": float(mse_dt),
            "validation_rmse": float(rmse_dt),
            "validation_mae": float(mae_dt),
            "validation_r2": float(r2_dt)
        }
        
        with open('2decision_tree_validation_metrics.json', 'w') as f:
            json.dump(metrics_dt, f, indent=4)
        print("决策树验证指标已保存到 2decision_tree_validation_metrics.json")
        
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred_dt, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('决策树模型: 预测值 vs 实际值')
        plt.savefig('2decision_tree_validation_plot.png')
        plt.close()
        print("决策树验证对比图已保存到 2decision_tree_validation_plot.png")
    except Exception as e:
        print(f"评估决策树模型时出错: {e}")
    
    # 比较所有模型
    try:
        models_comparison = {
            "XGBoost": {
                "RMSE": float(rmse_xgb),
                "MAE": float(mae_xgb),
                "R2": float(r2_xgb)
            }
        }
        
        if 'rmse_cb' in locals():
            models_comparison["CatBoost"] = {
                "RMSE": float(rmse_cb),
                "MAE": float(mae_cb),
                "R2": float(r2_cb)
            }
        
        if 'rmse_dt' in locals():
            models_comparison["DecisionTree"] = {
                "RMSE": float(rmse_dt),
                "MAE": float(mae_dt),
                "R2": float(r2_dt)
            }
        
        with open('models_validation_comparison.json', 'w') as f:
            json.dump(models_comparison, f, indent=4)
        print("\n所有模型的验证指标比较已保存到 models_validation_comparison.json")
        
        # 创建比较图表
        models = list(models_comparison.keys())
        rmse_values = [models_comparison[model]["RMSE"] for model in models]
        r2_values = [models_comparison[model]["R2"] for model in models]
        
        # RMSE比较图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(models, rmse_values)
        plt.title('模型RMSE比较 (越低越好)')
        plt.ylabel('RMSE')
        
        # R2比较图
        plt.subplot(1, 2, 2)
        plt.bar(models, r2_values)
        plt.title('模型R²比较 (越高越好)')
        plt.ylabel('R²')
        
        plt.tight_layout()
        plt.savefig('models_validation_comparison.png')
        plt.close()
        print("模型比较图已保存到 models_validation_comparison.png")
    except Exception as e:
        print(f"比较模型时出错: {e}")

if __name__ == "__main__":
    main()
