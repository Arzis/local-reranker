# 创建 Python 环境
````
conda create -n local-reranker python=3.11
conda activate local-reranker
````

# 自己安装cuda
````
https://blog.csdn.net/lyx2870657588/article/details/156237506
````

# PyTorch（GPU版，CUDA 12.1 为例）
````aiignore
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果只用 CPU
# pip install torch torchvision torchaudio

# 核心依赖
pip install transformers
pip install sentencepiece
pip install protobuf
````

# 下载模型
````aiignore
pip install modelscope

# 方法1：Python 下载
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='./models')"
````

# 使用 FastAPI 部署
```aiignore
pip install fastapi uvicorn
```

# 启动服务
```aiignore
python rerank_server.py
```
