"""一个零依赖的离线 Embedding 实现，仅用于教学演示。

为什么需要它？
--------------
DeepSeek 没有 Embedding 接口。为了让 RAG 课程能立刻跑起来、不额外引入
模型下载 / 网络依赖，这里用「词袋哈希」做一个确定性的伪向量。

它的缺点很明确：只认字面重合，不理解同义词。
生产环境请换成 langchain-openai / langchain-huggingface 的真实 embedding 模型，
其余代码（VectorStore、Retriever）完全不用改。
"""

from __future__ import annotations

import hashlib
import re

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _to_vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # 对中文按单字、对英文按单词切分（足够演示用）
        tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
        for token in tokens:
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[digest % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)
