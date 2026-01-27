from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model: str, device: str = 'cuda', max_length: int = 8192):
        self.model = SentenceTransformer(model, device=device)
        self.max_length = max_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts, max_length=self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode_query(text, max_length=self.max_length).tolist()