import numpy as np
import json
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import vstack
from tqdm import tqdm

from models.ML.ParliamentaryVectorization import ParliamentaryVectorization
from models.ML.PULKMeans import PULKMeans

class ParliamentaryClassifier:
    def __init__(self, subset_path: str, max_iter_pulk=10):
        self.subset_path = subset_path
        self.vectorizer_engine = ParliamentaryVectorization(subset_path)
        self.max_iter_pulk = max_iter_pulk
        self.results = {}

    def run_full_pipeline(self):
        # 1. Vectorización Global (Train)
        print("1. Vectorizando Corpus de Entrenamiento...")
        self.vectorizer_engine.load_and_vectorize()
        
        # Cargar datos de TEST para evaluar
        test_path = os.path.join(self.subset_path, "test.json")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        print("\n2. Iniciando entrenamiento por Diputado...")
        
        # Iteramos sobre cada diputado que tenga datos
        for mp_name in tqdm(self.vectorizer_engine.mp_indices.keys(), desc="Procesando Diputados"):
            #print(f"\n--- Procesando MP: {mp_name} ---")
            
            # A) Obtener matrices P (Positivos) y U (Unlabeled) de TRAIN
            try:
                P_train, U_train = self.vectorizer_engine.get_data_for_mp(mp_name)
            except ValueError:
                continue # Skip si hay error

            # B) Algoritmo PUL-KM: Detectar Negativos Fiables
            pul = PULKMeans(max_iter=self.max_iter_pulk) # 10 iteraciones suelen bastar
            rn_indices = pul.fit(P_train, U_train)
            
            # C) Construir Dataset de Entrenamiento Limpio
            # X_train = P + RN
            # y_train = 1 para P, 0 para RN
            RN_matrix = U_train[rn_indices]
            
            X_train = vstack([P_train, RN_matrix])
            y_train = np.concatenate([
                np.ones(P_train.shape[0]), # 1: Positivos (Intervenciones del MP)
                np.zeros(RN_matrix.shape[0]) # 0: Negativos Fiables
            ])
            
            # D) Entrenar SVM
            # class_weight='balanced' es CRÍTICO por el desbalance 1:14 que viste
            clf = LinearSVC(class_weight='balanced', random_state=42, dual='auto', max_iter=5000)
            clf.fit(X_train, y_train)
            
            # E) Evaluación en TEST
            # Necesitamos preparar el X_test y y_test para ESTE diputado
            # - Sus intervenciones en test son CLASE 1
            # - Las intervenciones de TODOS los demás en test son CLASE 0
            
            if mp_name not in test_data:
                print(f"   [Skip] No hay datos de test para {mp_name}")
                continue

            # Construir X_test dinámicamente
            mp_test_texts = test_data[mp_name]
            others_test_texts = []
            for other_mp, texts in test_data.items():
                if other_mp != mp_name:
                    others_test_texts.extend(texts)
            
            # Vectorizar test (usando el vectorizer ya entrenado)
            X_test_pos = self.vectorizer_engine.vectorizer.transform(mp_test_texts)
            X_test_neg = self.vectorizer_engine.vectorizer.transform(others_test_texts)
            
            X_test = vstack([X_test_pos, X_test_neg])
            y_test = np.concatenate([
                np.ones(X_test_pos.shape[0]),
                np.zeros(X_test_neg.shape[0])
            ])
            
            # Predicción
            y_pred = clf.predict(X_test)
            
            # Métricas
            # Nos interesa sobre todo la clase 1 (Positiva)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            
            # Guardamos resumen
            #print(f"   [RESULTADO] F1-Score (Clase MP): {f1:.4f}")
            #print(f"   (Train Size: {X_train.shape[0]} | Test Size: {X_test.shape[0]})")
            
            self.results[mp_name] = f1

        # Promedio final
        if self.results:
            avg_f1 = sum(self.results.values()) / len(self.results)
            print(f"\n=== FIN DEL PROCESO ===")
            print(f"F1-Score Promedio Global: {avg_f1:.4f}")
            return self.results
        else:
            return {}