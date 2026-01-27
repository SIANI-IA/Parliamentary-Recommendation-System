import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from models.IR.retriever import SparseRetriever, DenseRetriever, HybridRetriever
from eval.Evaluator import Evaluator

class ParliamentaryIRPipeline:
    def __init__(
            self, 
            dataset_path, 
            model_type="dense", 
            sparse_type="bm25", 
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ):

        assert model_type in ["sparse", "dense", "hybrid"], "model_type must be 'sparse', 'dense', or 'hybrid'"

        self.dataset_path = dataset_path
        self.model_type = model_type # 'sparse', 'dense', 'hybrid'
        self.sparse_type = sparse_type
        self.embedding_model_name = embedding_model
        
        # Cargar mapeo
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            self.mapping = json.load(f)
            self.id2label = {int(k): v for k, v in self.mapping['id2label'].items()}
            self.label2id = self.mapping['label2id']

    def _create_mp_profiles(self):
        """
        Crea documentos 'Macro' concatenando todas las intervenciones de cada MP en Train.
        Estrategia 'ir-p' del paper.
        """
        print("Creating MP Profiles from Train set...")
        train_ds = load_from_disk(self.dataset_path)['train']
        
        mp_texts = {mp_name: [] for mp_name in self.label2id.keys()}
        
        for row in tqdm(train_ds, desc="Aggregating texts"):
            speakers = row['Speakers']
            interventions = row['Interventions']
            
            for sp, texts in zip(speakers, interventions):
                if sp in mp_texts:
                    # Concatenamos con salto de línea
                    mp_texts[sp].extend(texts)
        
        # Crear objetos Document de LangChain
        docs = []
        for mp_name, text_list in mp_texts.items():
            if not text_list:
                continue # MP sin intervenciones (raro tras limpieza)
            
            full_text = "\n".join(text_list)
            # Metadata CRUCIAL: guardamos el ID numérico para la matriz de evaluación
            mp_id = self.label2id[mp_name]
            docs.append(Document(page_content=full_text, metadata={"mp_id": mp_id, "name": mp_name}))
            
        print(f"Created {len(docs)} MP profile documents.")
        return docs

    def run_evaluation(self, top_k: int = 10):
        # 1. Preparar Base de Datos (Indexing)
        print(f"\n=== FASE 1: Indexing ({self.model_type}) ===")
        docs = self._create_mp_profiles()
        
        # Embeddings (Necesarios para Dense y para inicializar FAISS en todos los casos)
        print(f"Loading embeddings: {self.embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        print("Building VectorStore (FAISS)...")
        # Incluso si usamos BM25, tu clase Retriever base pide un objeto FAISS
        db = FAISS.from_documents(docs, embeddings)
        
        # 2. Inicializar Retriever
        # Usamos un Top-K alto para llenar la matriz de scores con suficientes candidatos
        
        if self.model_type == "sparse":
            retriever_engine = SparseRetriever(db, self.sparse_type, top_k=top_k)
        elif self.model_type == "dense":
            retriever_engine = DenseRetriever(db, top_k=top_k)
        elif self.model_type == "hybrid":
            retriever_engine = HybridRetriever(db, self.sparse_type, top_k=top_k, alpha=0.5)
        else:
            raise ValueError("Unknown model type")
            
        # 3. Evaluación en Test
        print("\n=== FASE 2: Retrieval & Scoring ===")
        test_ds = load_from_disk(self.dataset_path)['test']
        X_test_texts = test_ds['Text']
        y_test_true = np.array(test_ds['label'])
        
        n_samples = len(X_test_texts)
        n_mps = len(self.label2id)
        
        # Matriz de Scores
        # Inicializamos en 0.0 (o un valor bajo). 
        # Los métodos de IR no dan scores para todos, solo para el Top-K.
        global_scores = np.zeros((n_samples, n_mps))
        
        for i, query_text in enumerate(tqdm(X_test_texts, desc="Querying")):
            # Retrieve
            retrieved_docs = retriever_engine.retrieve(query_text)
            
            # Asignar Scores basados en Ranking
            # LangChain (especialmente Ensemble/BM25) no siempre da un score numérico uniforme.
            # Una estrategia robusta para evaluación IR es Reciprocal Rank o Linear Decay.
            # Aquí usamos score = 1 / (rank + 1)
            
            for rank, doc in enumerate(retrieved_docs):
                mp_id = doc.metadata.get("mp_id")
                if mp_id is not None:
                    # Score de ranking: El 1º tiene score 1.0, el 2º 0.5, el 3º 0.33...
                    score = 1.0 / (rank + 1)
                    global_scores[i, mp_id] = score
        
        # 4. Calcular Métricas
        print("\n=== FASE 3: Computing Metrics ===")
        # threshold pequeño porque nuestros scores son bajos (0 a 1, decayendo rápido)
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        metrics = evaluator.compute_all_metrics(y_test_true, global_scores, threshold=0.1)
        evaluator.print_report(metrics)
        
        return metrics