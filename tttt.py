from safetensors.torch import load_file
import torch

model_path = r"./models/BAAI/bge-reranker-v2-m3"

print("正在读取 safetensors...")
state_dict = load_file(f"{model_path}/model.safetensors")
# 如果这一行就崩溃了，说明文件损坏，跳到第五步重新下载

print("正在保存为 pytorch_model.bin...")
torch.save(state_dict, f"{model_path}/pytorch_model.bin")
print("转换完成!")