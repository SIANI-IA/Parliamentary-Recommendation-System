import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class GlobalEvaluator:
    def __init__(self, mp_names, k_values=[1, 5, 10]):
        self.mp_names = mp_names
        self.k_values = k_values

    def compute_all_metrics(self, y_true_indices, y_scores_matrix):
        """
        Calcula métricas de clasificación (Micro/Macro) y de ranking (IR).
        """
        print(f"\n--- Evaluando {len(y_true_indices)} documentos de Test ---")
        metrics = {}
        
        # --- A. CLASIFICACIÓN (Top-1 Prediction) ---
        # Tomamos el diputado con mayor score como la predicción
        y_pred_indices = np.argmax(y_scores_matrix, axis=1)
        
        # 1. Accuracy
        metrics["Accuracy"] = accuracy_score(y_true_indices, y_pred_indices)
        
        # 2. Métricas Micro, Macro y Weighted
        # MICRO: Calcula globales contando total de aciertos/fallos (Importante para comparar con Paper)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average='micro', zero_division=0
        )
        
        # MACRO: Calcula métricas por diputado y luego promedia (sin importar si habló mucho o poco)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average='macro', zero_division=0
        )
        
        # WEIGHTED: Promedia pesando por número de intervenciones (útil para ver impacto real)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average='weighted', zero_division=0
        )
        
        # Guardamos todo
        metrics["Precision (Micro)"] = p_micro
        metrics["Recall (Micro)"]    = r_micro
        metrics["F1-Score (Micro)"]  = f1_micro
        
        metrics["Precision (Macro)"] = p_macro
        metrics["Recall (Macro)"]    = r_macro
        metrics["F1-Score (Macro)"]  = f1_macro
        
        metrics["Precision (Weighted)"] = p_weighted
        metrics["Recall (Weighted)"]    = r_weighted
        metrics["F1-Score (Weighted)"]  = f1_weighted

        # --- B. RANKING / IR (Igual que antes) ---
        mrr_sum = 0
        recalls_at_k = {k: 0 for k in self.k_values}
        ndcg_sum = 0
        n_samples = len(y_true_indices)
        
        for i in range(n_samples):
            true_idx = y_true_indices[i]
            scores = y_scores_matrix[i]
            
            # Ordenar descendente
            sorted_indices = np.argsort(scores)[::-1]
            
            # Buscar rango
            rank_tuple = np.where(sorted_indices == true_idx)[0]
            
            if len(rank_tuple) > 0:
                rank = rank_tuple[0] + 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / np.log2(rank + 1)
            else:
                rank = float('inf')

            for k in self.k_values:
                if rank <= k:
                    recalls_at_k[k] += 1

        metrics["MRR"] = mrr_sum / n_samples
        metrics["nDCG"] = ndcg_sum / n_samples
        for k in self.k_values:
            metrics[f"Recall@{k}"] = recalls_at_k[k] / n_samples

        return metrics

    def print_report(self, metrics):
        print("\n=== REPORTE DE EVALUACIÓN FINAL (COMPARABLE CON PAPER) ===")
        print(">> MÉTRICAS GLOBALES (MICRO)")
        print(f"   Accuracy:            {metrics['Accuracy']:.4f}")
        print(f"   Micro Precision:     {metrics['Precision (Micro)']:.4f}")
        print(f"   Micro Recall:        {metrics['Recall (Micro)']:.4f}")
        print(f"   Micro F1-Score:      {metrics['F1-Score (Micro)']:.4f}  <-- DATO CLAVE PAPER")
        
        print("\n>> MÉTRICAS PROMEDIO (MACRO)")
        print(f"   Macro Precision:     {metrics['Precision (Macro)']:.4f}")
        print(f"   Macro Recall:        {metrics['Recall (Macro)']:.4f}")
        print(f"   Macro F1-Score:      {metrics['F1-Score (Macro)']:.4f}  <-- DATO CLAVE PAPER")
        
        print("\n>> RANKING / IR")
        print(f"   MRR:                 {metrics['MRR']:.4f}")
        print(f"   nDCG:                {metrics['nDCG']:.4f}")
        for k in self.k_values:
            print(f"   Recall@{k:<2}:           {metrics[f'Recall@{k}']:.4f}")
        print("==========================================================")