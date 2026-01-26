import numpy as np
import json
import os
from sklearn.svm import LinearSVC
from scipy.sparse import vstack

from models.ML.ParliamentaryVectorization import ParliamentaryVectorization
from models.ML.PULKMeans import PULKMeans
from eval.Evaluator import Evaluator
from utils import SEED

class ParliamentaryClassifierGlobal:
    def __init__(self, subset_path, max_iter_pulk=10):
        self.subset_path = subset_path
        self.vectorizer_engine = ParliamentaryVectorization(subset_path)
        self.max_iter_pulk = max_iter_pulk
        
    def run_evaluation(self):
        # 1. Vectorización Global (usando TRAIN)
        print("1. Vectorizando y preparando datos...")
        self.vectorizer_engine.load_and_vectorize()
        
        # 2. Cargar y Vectorizar TODO el TEST set de una vez
        test_path = os.path.join(self.subset_path, "test.json")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data_json = json.load(f)
            
        # Lista ordenada de diputados (para asignar columnas en la matriz)
        # Solo consideramos diputados que existen en TRAIN (los que tienen modelo)
        mp_list = sorted(list(self.vectorizer_engine.mp_indices.keys()))
        mp_to_idx = {mp: i for i, mp in enumerate(mp_list)}
        
        # Aplanar Test Set
        # Necesitamos saber para cada documento: ¿Quién es su dueño real? (Índice)
        X_test_texts = []
        y_test_true_indices = []
        
        for mp, texts in test_data_json.items():
            if mp in mp_to_idx: # Solo evaluamos si el MP estaba en train
                idx = mp_to_idx[mp]
                for t in texts:
                    X_test_texts.append(t)
                    y_test_true_indices.append(idx)
        
        # Convertir textos de test a vectores (usando el vectorizer entrenado en train)
        print(f"   -> Vectorizando {len(X_test_texts)} documentos de test...")
        X_test_matrix = self.vectorizer_engine.vectorizer.transform(X_test_texts)
        y_test_true_indices = np.array(y_test_true_indices)
        
        # 3. Matriz Global de Scores
        # Filas: Documentos de Test
        # Columnas: Diputados (Modelos)
        # Inicializamos con un valor muy bajo (-1e9) por si algún modelo falla
        n_test_samples = X_test_matrix.shape[0]
        n_mps = len(mp_list)
        global_scores = np.full((n_test_samples, n_mps), -1e9)
        
        print(f"\n2. Entrenando {n_mps} modelos SVM (uno por diputado)...")
        
        for i, mp_name in enumerate(mp_list):
            print(f"   [{i+1}/{n_mps}] Procesando: {mp_name}", end="\r")
            
            # --- A. Obtener datos de TRAIN para este MP ---
            P_train, U_train = self.vectorizer_engine.get_data_for_mp(mp_name)
            
            # --- B. PUL: Detectar Negativos Fiables ---
            pul = PULKMeans(max_iter=self.max_iter_pulk)
            rn_indices = pul.fit(P_train, U_train) # El print interno de PUL puede ensuciar, puedes comentarlo
            
            # --- C. Armar Dataset de Entrenamiento ---
            RN_matrix = U_train[rn_indices]
            X_train = vstack([P_train, RN_matrix])
            y_train = np.concatenate([np.ones(P_train.shape[0]), np.zeros(RN_matrix.shape[0])])
            
            # --- D. SVM (LinearSVC) ---
            # Usamos class_weight='balanced'
            clf = LinearSVC(class_weight='balanced', random_state=SEED, dual='auto')
            clf.fit(X_train, y_train)
            
            # --- E. PREDICCIÓN GLOBAL ---
            # Obtenemos el score para TODOS los documentos de test
            # decision_function devuelve la distancia al hiperplano (con signo)
            # Mayor valor = Más probabilidad de ser de este diputado
            scores = clf.decision_function(X_test_matrix)
            
            # Guardamos en la columna correspondiente
            global_scores[:, i] = scores

        print("\n\n3. Calculando métricas finales...")
        evaluator = Evaluator(mp_list)
        metrics = evaluator.compute_all_metrics(y_test_true_indices, global_scores)
        evaluator.print_report(metrics)
        
        return metrics