import numpy as np
import os
import time
from sklearn.svm import LinearSVC
from scipy.sparse import vstack
from datasets import load_from_disk
from tqdm import tqdm

# Asegúrate de importar tus clases corregidas
from models.ML.ParliamentaryVectorization import ParliamentaryVectorization
from models.ML.PULKMeans import PULKMeans
from eval.Evaluator import Evaluator
from models.Recommender import Recommender

from utils import SEED

class PULKMRecommender(Recommender):
    def __init__(self, dataset_path: str, max_iter_pulk=10, lang='spanish'):
        super().__init__(dataset_path)
        self.vectorizer_engine = ParliamentaryVectorization(dataset_path, lang=lang)
        self.max_iter_pulk = max_iter_pulk
        self.best_threshold = 0.0 # Valor por defecto

    def __str__(self):
        return f"PULKMRecommender_maxiter{self.max_iter_pulk}"

    def _train_models(self, mp_list):
        """Entrena los N clasificadores SVM y devuelve una lista de modelos."""
        trained_models = []
        n_mps = len(mp_list)
        
        print(f"\n=== Entrenando {n_mps} Clasificadores SVM (One-vs-Rest) ===")
        
        for mp_id in tqdm(range(n_mps), desc="Entrenando"):
            # A. Obtener datos
            try:
                P_train, U_train = self.vectorizer_engine.get_data_for_mp(mp_id)
            except ValueError:
                trained_models.append(None) # Modelo fallido/vacío
                continue

            # B. PUL-KMeans
            pul = PULKMeans(max_iter=self.max_iter_pulk, verbose=False)
            rn_indices = pul.fit(P_train, U_train)
            
            # C. Dataset Binario
            if len(rn_indices) == 0:
                RN_matrix = U_train
            else:
                RN_matrix = U_train[rn_indices]
            
            X_train_bin = vstack([P_train, RN_matrix])
            y_train_bin = np.concatenate([np.ones(P_train.shape[0]), np.zeros(RN_matrix.shape[0])])
            
            # D. SVM
            clf = LinearSVC(class_weight='balanced', random_state=SEED, dual='auto')
            clf.fit(X_train_bin, y_train_bin)
            trained_models.append(clf)
            
        return trained_models

    def _predict_scores(self, models, X_matrix):
        """Genera la matriz de scores para un conjunto de datos dado."""
        n_samples = X_matrix.shape[0]
        n_mps = len(models)
        scores_matrix = np.zeros((n_samples, n_mps))
        
        # Para acelerar, podríamos paralelizar, pero el bucle simple es seguro
        for i, clf in enumerate(models):
            if clf is None:
                scores_matrix[:, i] = -100.0 # Penalización fuerte
            else:
                scores_matrix[:, i] = clf.decision_function(X_matrix)
        
        return scores_matrix

    def _optimize_threshold(self, y_true_dev, scores_dev):
        """Busca el mejor umbral maximizando Micro F1 en el conjunto DEV."""
        print("\n=== Optimizando Umbral en DEV ===")
        
        # LinearSVC devuelve distancias, no probs. El rango puede ser [-2, 2] aprox.
        # Probamos percentiles para no depender de la escala exacta
        # O un grid fijo alrededor de 0.0
        
        thresholds_to_test = np.linspace(-1.5, 1.5, 50) # Probamos 50 valores entre -1.5 y 1.5
        best_f1 = -1.0
        best_th = 0.0
                
        # Iteramos buscando el mejor F1 Micro
        for th in thresholds_to_test:
            # Calculamos métricas rápidas (sin imprimir reporte)
            # Solo necesitamos F1 Micro para decidir
            y_pred_bin = (scores_dev >= th).astype(int)
            
            # Cálculo manual rápido de F1 Micro para no sobrecargar
            # TP = (Pred=1 & True=1)
            tp = np.sum((y_pred_bin == 1) & (y_true_dev == 1))
            # FP = (Pred=1 & True=0)
            fp = np.sum((y_pred_bin == 1) & (y_true_dev == 0))
            # FN = (Pred=0 & True=1)
            fn = np.sum((y_pred_bin == 0) & (y_true_dev == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        print(f" -> Mejor Umbral encontrado: {best_th:.4f} (Micro F1 en Dev: {best_f1:.4f})")
        return best_th

    def run_full_pipeline(self):
        # 1. Vectorización (TRAIN)
        print("\n=== FASE 1: Vectorización ===")
        self.vectorizer_engine.load_and_vectorize()
        
        # Lista de MPs ordenada
        mp_list = [self.vectorizer_engine.id_to_name[i] for i in range(len(self.vectorizer_engine.id_to_name))]

        # 2. Entrenamiento
        print("\n=== FASE 2: Entrenamiento ===")
        models = self._train_models(mp_list)
        
        # 3. Validación (DEV) - Selección de Umbral
        print("\n=== FASE 3: Validación (DEV) ===")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        X_dev_texts = dev_ds['Text']
        y_dev_true = np.array(dev_ds['label'])
        
        X_dev_matrix = self.vectorizer_engine.vectorizer.transform(X_dev_texts)
        scores_dev = self._predict_scores(models, X_dev_matrix)
        
        # Elegir mejor umbral
        self.best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 4. Evaluación Final (TEST)
        print("\n=== FASE 4: Evaluación Final (TEST) ===")
        test_ds = load_from_disk(self.dataset_path)['test']
        X_test_texts = test_ds['Text']
        y_test_true = np.array(test_ds['label'])
        
        X_test_matrix = self.vectorizer_engine.vectorizer.transform(X_test_texts)
        scores_test = self._predict_scores(models, X_test_matrix)
        
        # Evaluar usando el umbral optimizado
        print(f"Aplicando umbral optimizado ({self.best_threshold:.4f}) en Test...")
        evaluator = Evaluator()
        metrics = evaluator.compute_all_metrics(y_test_true, scores_test, threshold=self.best_threshold)
        
        evaluator.print_report(metrics)

        dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
        folder_name = os.path.join(self.folder_to_save_results, dataset_name, self.__str__())
        self.save_artifacts(
            output_dir=folder_name,
            model_name=self.__str__(),
            y_true=y_test_true, 
            scores=scores_test, 
            threshold=self.best_threshold, 
            metrics=metrics,
            mp_names_ordered=mp_list
        )
        
        return metrics