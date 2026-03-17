import requests

response = requests.post("http://localhost:8100/rerank", json={
    "query": "什么是深度学习？",
    "documents": [
        "深度学习是机器学习的一个分支，使用多层神经网络。",
        "今天股市大涨，科技股表现亮眼。",
        "卷积神经网络常用于图像识别任务。",
        "Python 的 pandas 库可以处理数据。"
    ],
    "top_n": 3
})

for r in response.json()["results"]:
    print(f"Score: {r['relevance_score']:.4f} | [{r['index']}] {r['document']}")