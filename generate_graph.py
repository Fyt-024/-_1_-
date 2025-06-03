import os
import subprocess

# 生成DVC计算图的dot文件
subprocess.run("dvc dag --dot > dvc_graph.dot", shell=True)

# 生成详细的计算图
subprocess.run("dvc dag --dot --full > detailed_graph.dot", shell=True)

# 生成简洁的计算图
subprocess.run("dvc dag --dot --no-color > clean_graph.dot", shell=True)

# 将dot文件转换为PNG图像
# 检查是否安装了Graphviz
try:
    subprocess.run("dot -V", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 如果安装了Graphviz，则使用dot命令生成PNG
    subprocess.run("dot -Tpng dvc_graph.dot -o dvc_pipeline_graph.png", shell=True)
    print("DVC计算图已生成：dvc_pipeline_graph.png")
except subprocess.CalledProcessError:
    print("警告：未检测到Graphviz。无法生成PNG图像。")
    print("请安装Graphviz (https://graphviz.org/download/)，然后运行:")
    print("dot -Tpng dvc_graph.dot -o dvc_pipeline_graph.png")
