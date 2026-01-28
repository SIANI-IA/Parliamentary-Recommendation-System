import numpy as np
import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
from tqdm import tqdm

# Importamos tu evaluador
from eval.Evaluator import Evaluator
from models.Recommender import Recommender

class DenseEmbeddingRecommender(Recommender):
    def __init__(
            self, 
            dataset_path: str, 
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
            batch_size: int = 64,
            token_limit: int = 512,
            use_intervention_level: bool = False  # <--- NUEVO FLAG
        ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.use_intervention_level = use_intervention_level # True: Max Score (ir-i), False: Centroid (ir-p)
        
        # Configuración del dispositivo (CUDA si es posible)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Cargando modelo Dense en: {self.device.upper()} ---")
        print(f"--- Estrategia: {'INTERVENTION-LEVEL (Max Score)' if use_intervention_level else 'PROFILE-LEVEL (Centroid)'} ---")
        
        # Cargamos el modelo SBERT
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.model.max_seq_length = token_limit
        
        self.profiles_matrix = None 
        # Si es Centroid: (N_MPs, Dim)
        # Si es Intervention: (Total_Intervenciones, Dim)
        
        self.mp_to_rows_map = {} # Diccionario para mapear MP_ID -> [Indices de filas en profiles_matrix]
        
        # Cargar mapeo para asegurar orden
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.num_mps = len(self.id2label)

    def _create_profiles(self):
        """
        Crea los embeddings de referencia.
        - Si use_intervention_level=False: Calcula el CENTROIDE de cada MP.
        - Si use_intervention_level=True: Guarda TODOS los embeddings individuales.
        """
        print("1. Indexando intervenciones de MPs...")
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
        
        # Lista para acumular todos los vectores si usamos intervention-level
        all_embeddings_list = []
        self.mp_to_rows_map = {i: [] for i in range(self.num_mps)}
        
        embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Si es CENTROID, inicializamos matriz fija. Si es INTERVENTION, usaremos listas dinámicas.
        if not self.use_intervention_level:
            self.profiles_matrix = np.zeros((self.num_mps, embedding_dim), dtype=np.float32)

        current_row_idx = 0

        print(f"   Codificando (Batch Size: {self.batch_size})...")
        
        for i in tqdm(range(self.num_mps), desc="Generando Embeddings"):
            texts = mp_texts[i]
            if not texts:
                continue 
            
            # Codificamos sin convertir a numpy todavía para mantener tensores en GPU si es necesario
            # o convertimos a numpy directamente. Para consistencia con cos_sim, mejor Tensor si es cuda.
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
            
            if self.use_intervention_level:
                # ESTRATEGIA: MAX SCORE (Guardar todo)
                # Guardamos los vectores individuales
                all_embeddings_list.append(embeddings)
                
                # Registramos qué filas pertenecen a este MP
                num_new_rows = len(embeddings)
                self.mp_to_rows_map[i] = list(range(current_row_idx, current_row_idx + num_new_rows))
                current_row_idx += num_new_rows
                
            else:
                # ESTRATEGIA: CENTROIDE (Promediar)
                centroid = np.mean(embeddings, axis=0).astype(np.float32)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.profiles_matrix[i] = centroid

        if self.use_intervention_level:
            # Concatenar todos los embeddings en una gran matriz (Total_Ints, Dim)
            if all_embeddings_list:
                self.profiles_matrix = np.vstack(all_embeddings_list)
            else:
                self.profiles_matrix = np.zeros((1, embedding_dim), dtype=np.float32) # Fallback vacío
                
        print(f"   [OK] Índice creado. Shape: {self.profiles_matrix.shape}")

    def _compute_scores(self, query_embeddings):
        """
        Calcula la similitud entre queries y MPs.
        Maneja la lógica de 'Max Score' vs 'Cosine to Centroid'.
        """
        # 1. Calcular similitud contra TODA la matriz de perfiles
        # profiles_matrix puede ser (N_MPs, Dim) o (Total_Intervenciones, Dim)
        # scores_raw será (N_Queries, N_Rows_in_Profile)
        # Convertimos profiles_matrix a Tensor para usar util.cos_sim (optimizado en GPU)
        profiles_tensor = torch.tensor(self.profiles_matrix, device=self.device)
        queries_tensor = torch.tensor(query_embeddings, device=self.device) # Asegurar que es tensor
        
        raw_scores = util.cos_sim(queries_tensor, profiles_tensor)
        
        if not self.use_intervention_level:
            # Caso simple: Las columnas ya son los MPs (1 a 1)
            return raw_scores.cpu().numpy()
        
        else:
            # Caso complejo: Hay que agrupar (Group Max)
            # raw_scores shape: (N_Queries, Total_Intervenciones)
            # Queremos: (N_Queries, N_MPs)
            
            num_queries = raw_scores.shape[0]
            final_scores = torch.zeros((num_queries, self.num_mps), device=self.device) - 1.0 # Init con -1
            
            # Iteramos sobre los MPs para sacar su max score
            # Esto es más rápido que iterar sobre queries
            for mp_id in range(self.num_mps):
                row_indices = self.mp_to_rows_map.get(mp_id, [])
                if row_indices:
                    # Seleccionamos las columnas correspondientes a las intervenciones de este MP
                    # y tomamos el máximo valor a lo largo de esas columnas
                    mp_scores_all_interventions = raw_scores[:, row_indices]
                    max_scores, _ = torch.max(mp_scores_all_interventions, dim=1)
                    final_scores[:, mp_id] = max_scores
            
            return final_scores.cpu().numpy()

    def _optimize_threshold(self, y_true, scores):
        """Optimiza umbral en DEV basado en F1-Micro"""
        print("\n=== Optimizando Umbral en DEV ===")
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

    def run_evaluation(self):
        # 1. Crear Perfiles (TRAIN)
        self._create_profiles()
        
        # 2. Validación (DEV)
        print("\n2. Validando en DEV...")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        X_dev_emb = self.model.encode(dev_ds['Text'], batch_size=self.batch_size, show_progress_bar=True)
        y_dev_true = np.array(dev_ds['label'])
        
        # Usamos el nuevo método de cálculo
        scores_dev = self._compute_scores(X_dev_emb)
        
        best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 3. Test
        print("\n3. Evaluando en TEST...")
        test_ds = load_from_disk(self.dataset_path)['test']
        X_test_emb = self.model.encode(test_ds['Text'], batch_size=self.batch_size, show_progress_bar=True)
        y_test_true = np.array(test_ds['label'])
        
        scores_test = self._compute_scores(X_test_emb)
        
        # Evaluación
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        metrics = evaluator.compute_all_metrics(y_test_true, scores_test, threshold=best_threshold)
        evaluator.print_report(metrics)

        dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
        folder_name = os.path.join(self.folder_to_save_results, dataset_name, self.__str__())
        self.save_artifacts(
            output_dir=folder_name, 
            y_true=y_test_true, 
            scores=scores_test, 
            threshold=best_threshold, 
            metrics=metrics
        )
        
        return metrics