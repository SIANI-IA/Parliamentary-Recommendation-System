import numpy as np
import os
import json
import pickle
from datasets import load_from_disk
from tqdm import tqdm
import nltk

# LangChain imports
from langchain_community.retrievers import BM25Retriever

# Imports de tu proyecto
from eval.Evaluator import Evaluator
from models.Recommender import Recommender

class SparseBM25Recommender(Recommender):
    def __init__(self, dataset_path: str, lang: str = "spanish"):
        super().__init__(dataset_path)
        self.retriever = None
        self.mp_names_ordered = []  # Para saber quién es la fila i
        self.lang = lang
        
        # Cargar mapeo para asegurar el orden correcto de los IDs
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.num_mps = len(self.id2label)

    def __str__(self):
        return f"SparseBM25_ReciprocalRank"

    def _create_profiles(self):
        """
        Crea un 'Macro-Documento' por diputado concatenando todo su TRAIN.
        Inicializa el BM25Retriever de LangChain con la configuración por defecto.
        """
        print("1. Construyendo Perfiles de Diputados...")
        train_ds = load_from_disk(self.dataset_path)['train']
        
        # Diccionario acumulador: {mp_id: [lista_textos]}
        mp_texts = {i: [] for i in range(self.num_mps)}
        
        for row in tqdm(train_ds, desc="Agrupando textos"):
            speakers = row['Speakers']
            interventions = row['Interventions']
            
            label2id = {v: k for k, v in self.id2label.items()}
            
            for sp, texts in zip(speakers, interventions):
                if sp in label2id:
                    mid = label2id[sp]
                    mp_texts[mid].extend(texts)
        
        # Crear corpus ordenado por ID
        corpus_profiles = []
        for i in range(self.num_mps):
            full_text = " ".join(mp_texts[i])
            corpus_profiles.append(full_text)
            self.mp_names_ordered.append(self.id2label[i])

        print("2. Indexando con LangChain BM25Retriever (Default)...")
        
        # Instanciamos el Retriever SIN preprocess_func (usa el default)
        self.retriever = BM25Retriever.from_texts(corpus_profiles)
        
        # Establecemos k igual al total de MPs para poder recuperar scores de todos
        self.retriever.k = self.num_mps
        
        print(f"   [OK] Índice BM25 creado con {self.num_mps} perfiles.")

    def _compute_scores_matrix(self, text_list):
        """
        Calcula la matriz de similitud usando BM25 y transforma los scores
        usando Reciprocal Rank Decay: Score = 1 / (Rank + 1).
        """
        scores_matrix = []
        
        # El objeto interno rank_bm25
        bm25_obj = self.retriever.vectorizer
        
        for text in tqdm(text_list, desc="Calculando BM25 Scores"):
            # NOTA: Aunque usemos el default, rank_bm25 necesita recibir una lista de tokens.
            # Hacemos el split más básico posible (.lower().split()) para que funcione la librería.
            tokenized_query = text.lower().split()
            
            # 1. Obtenemos scores crudos de BM25 (ej: [10.5, 2.1, 40.0 ...])
            raw_scores = bm25_obj.get_scores(tokenized_query)
            raw_scores = np.array(raw_scores)
            
            # 2. Convertimos Scores a Ranks
            # argsort devuelve los índices ordenados de menor a mayor.
            # Hacemos [::-1] para tener de mayor a menor (el mejor score primero).
            sorted_indices = np.argsort(raw_scores)[::-1]
            
            # 3. Asignar Score basado en Rank: 1 / (rank + 1)
            # Creamos un array vacío del tamaño de MPs
            rank_decay_scores = np.zeros_like(raw_scores, dtype=float)
            
            # Llenamos el array: 
            # El índice que quedó 1º (rank 0) recibe 1/1 = 1.0
            # El índice que quedó 2º (rank 1) recibe 1/2 = 0.5
            for rank, mp_idx in enumerate(sorted_indices):
                rank_decay_scores[mp_idx] = 1.0 / (rank + 1)
                
            scores_matrix.append(rank_decay_scores)
            
        return np.array(scores_matrix)

    def _optimize_threshold(self, y_true, scores):
        """Busca el mejor umbral en DEV para maximizar Micro-F1"""
        print("\n=== Optimizando Umbral en DEV (Reciprocal Rank) ===")
        
        # Dado que los scores ahora son 1, 0.5, 0.33, 0.25...
        # Los umbrales deben ser muy bajos para capturar algo más allá del Top-1 o Top-2.
        # Ajustamos el linspace para buscar con precisión en la parte baja.
        thresholds = np.concatenate([
            np.linspace(0.001, 0.2, 50), # Zona de ranks bajos (Top 5-1000)
            np.linspace(0.25, 1.0, 10)   # Zona de ranks altos (Top 1-4)
        ])
        thresholds = np.unique(thresholds) # Quitar duplicados y ordenar
        
        best_f1 = -1.0
        best_th = 0.05
        
        for th in thresholds:
            y_pred_bin = (scores >= th).astype(int)
            
            tp = np.sum((y_pred_bin == 1) & (y_true == 1))
            fp = np.sum((y_pred_bin == 1) & (y_true == 0))
            fn = np.sum((y_pred_bin == 0) & (y_true == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        print(f" -> Mejor Umbral (Rank Decay): {best_th:.4f} (F1: {best_f1:.4f})")
        return best_th

    def run_evaluation(self):
        # 1. Train (Indexar)
        self._create_profiles()
        
        # 2. Validación (Dev)
        print("\n3. Validando en DEV...")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        y_dev_true = np.array(dev_ds['label'])
        
        # Calcular scores BM25 -> Convertidos a 1/(rank+1)
        scores_dev = self._compute_scores_matrix(dev_ds['Text'])
        
        best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 3. Test
        print("\n4. Evaluando en TEST...")
        test_ds = load_from_disk(self.dataset_path)['test']
        y_test_true = np.array(test_ds['label'])
        
        scores_test = self._compute_scores_matrix(test_ds['Text'])
        
        # Evaluación
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        metrics = evaluator.compute_all_metrics(y_test_true, scores_test, threshold=best_threshold)
        evaluator.print_report(metrics)

        # Guardado
        dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
        folder_name = os.path.join(self.folder_to_save_results, dataset_name, self.__str__())
        
        self.save_artifacts(
            output_dir=folder_name,
            model_name=self.__str__(),
            y_true=y_test_true, 
            scores=scores_test, 
            threshold=best_threshold, 
            metrics=metrics,
            mp_names_ordered=self.mp_names_ordered,
        )
        
        return metrics