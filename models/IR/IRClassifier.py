import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

from models.ML.ParliamentaryVectorization import ParliamentaryVectorization
from models.ML.PULKMeans import PULKMeans
from eval.Evaluator import Evaluator
from utils import SEED

class IRClassifier:
    def __init__(self, subset_path):
        self.subset_path = subset_path
        self.vectorizer_engine = ParliamentaryVectorization(subset_path)
        self.mp_profiles = {} # Diccionario: {mp_name: vector_centroide}
        self.mp_list = []      # Lista ordenada para mantener consistencia en la matriz

    def run_evaluation(self):
        print(f"--- INICIANDO BASELINE IR (Vector Space Model) ---")
        
        # 1. Vectorización Global (Train)
        print("1. Vectorizando Corpus de Entrenamiento...")
        self.vectorizer_engine.load_and_vectorize()
        
        # 2. Construir PERFILES de Diputados (Centroides)
        print("2. Construyendo perfiles vectoriales (Centroides)...")
        # Obtenemos la lista de diputados ordenada
        self.mp_list = sorted(list(self.vectorizer_engine.mp_indices.keys()))
        
        # Pre-reservamos una matriz para los perfiles: (N_Diputados, N_Features)
        n_features = self.vectorizer_engine.tfidf_matrix.shape[1]
        profiles_matrix = np.zeros((len(self.mp_list), n_features))
        
        for i, mp_name in enumerate(self.mp_list):
            # Obtenemos sus documentos de Train
            P_indices = self.vectorizer_engine.mp_indices[mp_name]
            P_matrix = self.vectorizer_engine.tfidf_matrix[P_indices]
            
            # Calculamos el CENTROIDE (promedio de sus vectores TF-IDF)
            # Esto representa el "tema promedio" o "vocabulario medio" del diputado
            centroid = np.array(P_matrix.mean(axis=0)).flatten()
            profiles_matrix[i] = centroid
            
        print(f"   [OK] Perfiles creados para {len(self.mp_list)} diputados.")

        # 3. Preparar Datos de Test
        test_path = os.path.join(self.subset_path, "test.json")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data_json = json.load(f)
            
        X_test_texts = []
        y_test_true_indices = []
        mp_to_idx = {mp: i for i, mp in enumerate(self.mp_list)}
        
        for mp, texts in test_data_json.items():
            if mp in mp_to_idx:
                idx = mp_to_idx[mp]
                for t in texts:
                    X_test_texts.append(t)
                    y_test_true_indices.append(idx)
        
        print(f"3. Vectorizando {len(X_test_texts)} documentos de Test...")
        # Usamos el mismo vectorizador de train para transformar test
        X_test_matrix = self.vectorizer_engine.vectorizer.transform(X_test_texts)
        y_test_true_indices = np.array(y_test_true_indices)

        # 4. Cálculo de Similitud (Matriz Global de Scores)
        print("4. Calculando Similitud Coseno (Query vs Profiles)...")
        # cosine_similarity devuelve una matriz (n_samples_X, n_samples_Y)
        # Aquí: (Docs_Test, Diputados) -> Exactamente lo que necesitamos
        global_scores = cosine_similarity(X_test_matrix, profiles_matrix)
        
        # 5. Evaluación
        print("\n5. Calculando métricas...")
        evaluator = Evaluator(self.mp_list)
        metrics = evaluator.compute_all_metrics(y_test_true_indices, global_scores)
        evaluator.print_report(metrics)
        
        return metrics