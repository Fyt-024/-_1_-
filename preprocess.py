
import pandas as pd

def main():
    # 读取原始数据
    data = pd.read_csv("filtered_data_with_median.csv")
    
    # 选择需要的列：'selling_price', 'torque', 'max_power', 'km_driven'
    data = data[['selling_price', 'torque', 'max_power', 'km_driven']]
    
    # 删除包含 NaN 的行
    data = data.dropna()
    
    # 保存预处理后的数据
    data.to_csv("preprocessed_data.csv", index=False)
    print("预处理完成，数据已保存到 preprocessed_data.csv")

if __name__ == "__main__":
    main()
