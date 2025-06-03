import os

# 生成详细的DVC计算图
os.system("dvc dag --dot > detailed_graph.dot")

# 生成简洁的DVC计算图
os.system("dvc dag --dot --full > clean_graph.dot")

# 生成完整的DVC图
os.system("dvc pipeline show --dot > dvc_graph.dot")

# 将dot文件转换为PNG图像
os.system("dot -Tpng detailed_graph.dot -o detailed_graph.png")
os.system("dot -Tpng clean_graph.dot -o clean_graph.png")
os.system("dot -Tpng dvc_graph.dot -o dvc_pipeline_graph.png")

print("DVC计算图已生成")
