import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ===== 1. 加载模型 =====
model_path = "./models/BAAI/bge-reranker-v2-m3"  # 本地路径

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"模型已加载到: {device}")

# ===== 2. 准备数据 =====
# 格式: [(query, document), ...]
pairs = [
    ("什么是机器学习？", "机器学习是人工智能的一个分支，通过数据训练模型来做预测。"),
    ("什么是机器学习？", "今天天气很好，适合出去散步。"),
    ("什么是机器学习？", "深度学习是机器学习的子领域，使用神经网络进行学习。"),
    ("什么是机器学习？", "Python是一种流行的编程语言。"),
]

# ===== 3. 计算相关性分数 =====
with torch.no_grad():
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(device)

    scores = model(**inputs, return_dict=True).logits.view(-1).float()
    scores = scores.cpu().tolist()

# ===== 4. 输出结果 =====
for pair, score in sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True):
    print(f"Score: {score:.4f} | {pair[1][:50]}")