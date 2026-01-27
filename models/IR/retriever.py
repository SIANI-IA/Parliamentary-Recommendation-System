from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import (
    BM25Retriever, 
    TFIDFRetriever,
)
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


class Retriever(ABC):
    
    def __init__(self, db: FAISS, top_k: int = 5):
        self.db = db
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str):
        pass


class SparseRetriever(Retriever):
    
    def __init__(self, db: FAISS, sparse_retriever: str, top_k: int = 5):
        
        assert sparse_retriever in ["bm25", "tfidf"], "sparse_retriever must be 'bm25' or 'tfidf'"
        
        super().__init__(db, top_k=top_k)
        all_docs = list(self.db.docstore._dict.values())
        
        if sparse_retriever == "bm25":
            self.sparse_retriever = BM25Retriever.from_documents(all_docs)
        elif sparse_retriever == "tfidf":
            self.sparse_retriever = TFIDFRetriever.from_documents(all_docs)
        
        self.sparse_retriever.k = top_k

    def __str__(self):
        return f"SparseRetriever(top_k={self.top_k}, type='{self.sparse_retriever.__class__.__name__}')"

    def retrieve(self, query: str):
        return self.sparse_retriever.invoke(query)


class DenseRetriever(Retriever):

    def __init__(self, db: FAISS, top_k: int = 5, search_type: str = "similarity", lambda_mult: float = 0.5):
        
        assert search_type in ["similarity", "mmr"], "search_type must be 'similarity' or 'mmr'(maximal marginal relevance)"
        assert (search_type == "similarity") or (lambda_mult is not None and 0 <= lambda_mult <= 1), f"lambda_mult must be between 0 and 1 when using mmr, not {lambda_mult}"
        
        super().__init__(db, top_k=top_k)
        self.search_type = search_type
        self.lambda_mult = lambda_mult
        search_kwargs = {"k": top_k} if search_type == "similarity" else {"k": top_k, "fetch_k": top_k*2, "lambda_mult": lambda_mult}
        self.dense_retriver = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def __str__(self):
        if self.search_type == "similarity":
            return f"NaiveDenseRetriever(top_k={self.top_k}, search_type='{self.search_type}')"
        else:
            return f"NaiveDenseRetriever(top_k={self.top_k}, search_type='{self.search_type}', lambda_mult={self.lambda_mult})"

    def retrieve(self, query: str):
        return self.dense_retriver.invoke(query)
    

class HybridRetriever(DenseRetriever):


    def _create_sparse_retriever(self, sparse_retriever: str):
        all_docs = list(self.db.docstore._dict.values())
        if sparse_retriever == "bm25":
            return BM25Retriever.from_documents(all_docs)
        elif sparse_retriever == "tfidf":
            return TFIDFRetriever.from_documents(all_docs)
        else:
            raise ValueError("sparse_retriever must be 'bm25' or 'tfidf'")

    def __init__(
            self, 
            db: FAISS, 
            sparse_retriever_name: str, 
            top_k: int = 5, 
            alpha: float = 0.7,
            search_type: str = "similarity",
            lambda_mult: float = 0.5
        ):
        
        super().__init__(db, top_k=top_k, search_type=search_type, lambda_mult=lambda_mult)

        assert sparse_retriever_name in ["bm25", "tfidf"], "sparse_retriever must be 'bm25' or 'tfidf'"
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
        
        self.alpha = alpha
        self.sparse_retriever_name = sparse_retriever_name
        self.sparse_retriever = self._create_sparse_retriever(sparse_retriever_name)
        self.sparse_retriever.k = top_k

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriver, self.sparse_retriever],
            weights=[self.alpha, 1 - self.alpha]
        )

    def __str__(self):
        if self.search_type == "similarity":
            return f"HybridRetriever(top_k={self.top_k}, search_type='{self.search_type}', alpha={self.alpha}, sparse_retriever={self.sparse_retriever_name})"
        else:
            return f"HybridRetriever(top_k={self.top_k}, search_type='{self.search_type}', lambda_mult={self.lambda_mult}, alpha={self.alpha}, sparse_retriever={self.sparse_retriever_name})"

    def retrieve(self, query: str):
        return self.ensemble_retriever.invoke(query)
    
# -----------------------------
# Example usage in other parts of the codebase:
# -----------------------------
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    # Example: Initialize a FAISS DB and use the retrievers
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda"}
    )
    docs = [
        Document(page_content="Paris is the capital of France.", metadata={"id": 1}),
        Document(page_content="Berlin is the capital of Germany.", metadata={"id": 2}),
        Document(page_content="Madrid is the capital of Spain.", metadata={"id": 3}),
        Document(page_content="Rome is the capital of Italy.", metadata={"id": 4}),
        Document(page_content="Lisbon is the capital of Portugal.", metadata={"id": 5}),
        Document(page_content="Vienna is the capital of Austria.", metadata={"id": 6}),
        Document(page_content="Brussels is the capital of Belgium.", metadata={"id": 7}),
        Document(page_content="Amsterdam is the capital of the Netherlands.", metadata={"id": 8}),
        Document(page_content="Copenhagen is the capital of Denmark.", metadata={"id": 9}),
        Document(page_content="Oslo is the capital of Norway.", metadata={"id": 10}),
    ]
    db = FAISS.from_documents(docs, embedding_model)

    # Sparse Retriever
    TOP_K = 5
    print("Sparse Retriever Results:")
    sparse_retriever = SparseRetriever(db, sparse_retriever="bm25", top_k=TOP_K)
    print(sparse_retriever)
    results = sparse_retriever.retrieve("What is the capital of France?")
    for doc in results:
        print(doc.page_content, doc.metadata)

    # Naive Dense Retriever
    TOP_K = 5
    print("Naive Dense Retriever Results:")
    dense_retriever = DenseRetriever(db, top_k=TOP_K, search_type="similarity")
    print(dense_retriever)
    results = dense_retriever.retrieve("What is the capital of France?")
    for doc in results:
        print(doc.page_content, doc.metadata)

    # Hybrid Retriever Example
    print("\nHybrid Retriever Results:")
    hybrid_retriever = HybridRetriever(db, "tfidf", top_k=TOP_K, alpha=0.7)
    print(hybrid_retriever)
    results = hybrid_retriever.retrieve("What is the capital of France?")
    for doc in results:
        print(doc.page_content, doc.metadata)
        