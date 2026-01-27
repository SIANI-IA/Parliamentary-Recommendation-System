import numpy as np
import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
from tqdm import tqdm

# Importamos tu evaluador
from eval.Evaluator import Evaluator

class ManualDensePipeline:
    def __init__(
            self, 
            dataset_path: str, 
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
            batch_size: int = 64,
            token_limit: int = 512,
        ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
        # Configuración del dispositivo (CUDA si es posible)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Cargando modelo Dense en: {self.device.upper()} ---")
        
        # Cargamos el modelo SBERT
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.model.max_seq_length = token_limit
        
        self.profiles_matrix = None # Matriz (N_MPs, Dimensión_Embedding)
        
        # Cargar mapeo para asegurar orden
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.num_mps = len(self.id2label)

    def _create_profiles(self):
        """
        Crea perfiles calculando el CENTROIDE (media) de los embeddings de cada MP.
        Estrategia 'ir-c' (Centroid-based), ideal para Dense Retrieval.
        """
        print("1. Construyendo Perfiles Densos (Estrategia Centroid)...")
        train_ds = load_from_disk(self.dataset_path)['train']
        
        # Agrupar textos por MP ID
        mp_texts = {i: [] for i in range(self.num_mps)}
        label2id = {v: k for k, v in self.id2label.items()}
        
        for row in tqdm(train_ds, desc="Agrupando textos"):
            speakers = row['Speakers']
            interventions = row['Interventions']
            for sp, texts in zip(speakers, interventions):
                if sp in label2id:
                    mid = label2id[sp]
                    mp_texts[mid].extend(texts)
        
        # Calcular Centroides
        # Dimensiones del modelo (ej. 384 para MiniLM)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.profiles_matrix = np.zeros((self.num_mps, embedding_dim), dtype=np.float32)
        
        print(f"   Codificando intervenciones por diputado (Batch Size: {self.batch_size})...")
        
        for i in tqdm(range(self.num_mps), desc="Generando Perfiles"):
            texts = mp_texts[i]
            if not texts:
                continue # MP sin textos, se queda en 0.0
            
            # Codificar todas las intervenciones de este MP
            # show_progress_bar=False para no ensuciar el log general
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
            
            # Calcular la media (Centroide)
            centroid = np.mean(embeddings, axis=0).astype(np.float32)
            
            # Normalizar el vector para que la similitud coseno funcione mejor
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
            self.profiles_matrix[i] = centroid
            
        print(f"   [OK] Matriz de Perfiles creada. Shape: {self.profiles_matrix.shape}")

    def _optimize_threshold(self, y_true, scores):
        """Optimiza umbral en DEV basado en F1-Micro"""
        print("\n=== Optimizando Umbral en DEV ===")
        # Dense scores van de -1 a 1, pero suelen concentrarse entre 0.2 y 0.8
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = -1.0
        best_th = 0.5
        
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
                
        print(f" -> Mejor Umbral: {best_th:.4f} (F1: {best_f1:.4f})")
        return best_th

    def run_full_evaluation(self):
        # 1. Crear Perfiles (TRAIN)
        self._create_profiles()
        
        # 2. Validación (DEV)
        print("\n2. Validando en DEV...")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        # Codificamos las Queries (Iniciativas completas)
        X_dev_emb = self.model.encode(dev_ds['Text'], batch_size=self.batch_size, show_progress_bar=True)
        y_dev_true = np.array(dev_ds['label'])
        
        # Similitud Coseno (usando util.cos_sim de sbert o numpy)
        # util.cos_sim devuelve Tensor, convertimos a numpy
        scores_dev = util.cos_sim(X_dev_emb, self.profiles_matrix).cpu().numpy()
        
        best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 3. Test
        print("\n3. Evaluando en TEST...")
        test_ds = load_from_disk(self.dataset_path)['test']
        X_test_emb = self.model.encode(test_ds['Text'], batch_size=self.batch_size, show_progress_bar=True)
        y_test_true = np.array(test_ds['label'])
        
        scores_test = util.cos_sim(X_test_emb, self.profiles_matrix).cpu().numpy()
        
        # Evaluación
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        metrics = evaluator.compute_all_metrics(y_test_true, scores_test, threshold=best_threshold)
        evaluator.print_report(metrics)
        
        return metrics