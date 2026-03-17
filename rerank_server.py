import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional

# ===== 初始化 =====
app = FastAPI(title="BGE Reranker API")

MODEL_PATH = "./models/BAAI/bge-reranker-v2-m3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 1024
BATCH_SIZE = 32

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print(f"模型加载完成，设备: {DEVICE}")


# ===== 数据模型 =====
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = None  # 返回前N个结果


class RerankResult(BaseModel):
    index: int
    document: str
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# ===== 核心重排函数 =====
def compute_scores(query: str, documents: List[str]) -> List[float]:
    pairs = [(query, doc) for doc in documents]
    all_scores = []

    # 分批处理，避免显存溢出
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
            all_scores.extend(scores.cpu().tolist())

    return all_scores


# ===== API 端点 =====
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    scores = compute_scores(request.query, request.documents)

    # 组装结果并排序
    results = [
        RerankResult(index=i, document=doc, relevance_score=score)
        for i, (doc, score) in enumerate(zip(request.documents, scores))
    ]
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    # 截取 top_n
    if request.top_n:
        results = results[:request.top_n]

    return RerankResponse(results=results)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)