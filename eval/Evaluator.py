import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, ndcg_score

class Evaluator:
    def __init__(self, k_values=[1, 5, 10, 20]):
        self.k_values = k_values

    def compute_all_metrics(self, y_true_matrix, y_scores_matrix, threshold=0.5):
        """
        Calcula métricas para Multi-Label Classification y Ranking IR.
        
        Args:
            y_true_matrix: Matriz binaria (N_muestras x N_clases) con el Ground Truth.
            y_scores_matrix: Matriz de floats (N_muestras x N_clases) con probabilidades/scores.
            threshold: Umbral para convertir scores en predicciones binarias (Clasificación).
        """
        print(f"\n--- Evaluando {y_true_matrix.shape[0]} documentos de Test (Multi-Label) ---")
        metrics = {}
        
        # --- A. CLASIFICACIÓN (Threshold-based) ---
        # Convertimos scores continuos a predicciones binarias (0 o 1)
        y_pred_binary = (y_scores_matrix >= threshold).astype(int)
        
        # 1. Exact Match Ratio (Subset Accuracy)
        # Es muy estricta: exige acertar TODOS los oradores y NO fallar ninguno
        metrics["Accuracy (Subset)"] = accuracy_score(y_true_matrix, y_pred_binary)
        
        # 2. Métricas Micro/Macro/Weighted (Standard en Papers ML)
        # MICRO: Agrega TP, FP, FN globalmente (importante para clases desbalanceadas)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_matrix, y_pred_binary, average='micro', zero_division=0
        )
        
        # MACRO: Promedio aritmético de métricas por clase
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_matrix, y_pred_binary, average='macro', zero_division=0
        )
        
        metrics["Precision (Micro)"] = p_micro
        metrics["Recall (Micro)"]    = r_micro
        metrics["F1-Score (Micro)"]  = f1_micro
        
        metrics["Precision (Macro)"] = p_macro
        metrics["Recall (Macro)"]    = r_macro
        metrics["F1-Score (Macro)"]  = f1_macro

        # --- B. RANKING / IR (List-wise) ---
        
        # 1. MAP (Mean Average Precision)
        # average='samples' calcula la AP para cada fila y hace la media -> Esto es MAP
        metrics["MAP"] = average_precision_score(y_true_matrix, y_scores_matrix, average='samples')
        
        # 2. nDCG (Normalized Discounted Cumulative Gain)
        # Sklearn lo calcula eficientemente para matrices
        metrics["nDCG"] = ndcg_score(y_true_matrix, y_scores_matrix)

        # 3. Recall@K (Multi-label)
        # Definición: De todos los oradores reales, ¿qué fracción aparece en el Top-K predicho?
        n_samples = y_true_matrix.shape[0]
        recalls_at_k = {k: 0.0 for k in self.k_values}
        
        for i in range(n_samples):
            true_indices = np.where(y_true_matrix[i] == 1)[0]
            if len(true_indices) == 0:
                continue # Evitar división por cero si no hay oradores (no debería pasar tras limpieza)
                
            # Obtener los índices de los K scores más altos
            # argsort ordena ascendente, tomamos el final y damos la vuelta
            scores = y_scores_matrix[i]
            sorted_indices = np.argsort(scores)[::-1]
            
            for k in self.k_values:
                top_k_indices = sorted_indices[:k]
                
                # Intersección: cuántos de mis Top-K son reales
                hits = len(set(top_k_indices) & set(true_indices))
                
                # Recall@K = Hits / Total_Relevantes
                # Nota: Si k < Total_Relevantes, el recall maximo es k/Total
                recalls_at_k[k] += hits / len(true_indices)

        for k in self.k_values:
            metrics[f"Recall@{k}"] = recalls_at_k[k] / n_samples

        return metrics

    def print_report(self, metrics):
        print("\n=== REPORTE DE EVALUACIÓN MULTI-LABEL ===")
        print(">> CLASIFICACIÓN (Threshold-based)")
        print(f"   Accuracy (Subset):   {metrics['Accuracy (Subset)']:.4f}")
        print(f"   Micro F1-Score:      {metrics['F1-Score (Micro)']:.4f}  <-- Important for classification")
        print(f"   Macro F1-Score:      {metrics['F1-Score (Macro)']:.4f}")
        
        print("\n>> RANKING / IR")
        print(f"   MAP:                 {metrics['MAP']:.4f}  <-- Important for ranking")
        print(f"   nDCG:                {metrics['nDCG']:.4f}")
        for k in self.k_values:
            print(f"   Recall@{k:<2}:           {metrics[f'Recall@{k}']:.4f}")
        print("=========================================")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Simulación de datos
    y_true = np.array([[1, 1, 1], [0, 1, 0]]) # 2 documentos, 3 oradores
    y_scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.6, 0.3]]) 
    
    ev = Evaluator()
    m = ev.compute_all_metrics(y_true, y_scores, threshold=0.5)
    ev.print_report(m)