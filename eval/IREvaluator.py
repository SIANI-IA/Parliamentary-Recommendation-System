import numpy as np
from sklearn.metrics import ndcg_score

class IREvaluator:
    def __init__(self, k_values=[1, 3, 5, 10, 20]):
        self.k_values = k_values

    def compute_metrics(self, y_true_indices, y_scores_matrix):
        """
        y_true_indices: Lista de enteros. El índice del MP correcto para cada documento.
                        Forma: (n_samples, )
        y_scores_matrix: Matriz de floats. Scores asignados a cada MP.
                        Forma: (n_samples, n_classes/mps)
        """
        n_samples = len(y_true_indices)
        metrics = {
            "MRR": 0.0,
            "nDCG": 0.0,
        }
        for k in self.k_values:
            metrics[f"Recall@{k}"] = 0.0

        print(f"Evaluando {n_samples} documentos...")

        # Convertir a numpy para velocidad
        y_scores = np.array(y_scores_matrix)
        
        mrr_sum = 0
        
        # Iteramos documento a documento
        for i in range(n_samples):
            true_idx = y_true_indices[i]
            scores = y_scores[i]
            
            # Ordenar índices de mayor a menor score
            # argsort ordena ascendente, así que invertimos
            sorted_indices = np.argsort(scores)[::-1]
            
            # 1. Calcular Rango (Rank) donde aparece el verdadero
            # np.where devuelve una tupla, tomamos el primer elemento
            rank_array = np.where(sorted_indices == true_idx)[0]
            
            if len(rank_array) > 0:
                rank = rank_array[0] + 1 # +1 porque es 1-based index
                mrr_sum += 1.0 / rank
            else:
                rank = float('inf') # No debería pasar si están todos los MPs

            # 2. Calcular Recall@k
            for k in self.k_values:
                if rank <= k:
                    metrics[f"Recall@{k}"] += 1

        # Promedios
        metrics["MRR"] = mrr_sum / n_samples
        for k in self.k_values:
            metrics[f"Recall@{k}"] /= n_samples
            
        # 3. Calcular nDCG (Scikit-learn lo hace vectorizado, pero requiere one-hot en y_true)
        # Para simplificar y no gastar memoria creando una matriz one-hot gigante,
        # usamos la lógica simplificada: nDCG = 1 / log2(rank + 1) en single-label
        ndcg_sum = 0
        for i in range(n_samples):
            true_idx = y_true_indices[i]
            scores = y_scores[i]
            sorted_indices = np.argsort(scores)[::-1]
            rank = np.where(sorted_indices == true_idx)[0][0] + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)
            
        metrics["nDCG"] = ndcg_sum / n_samples

        return metrics

# --- EJEMPLO DE USO FICTICIO ---
if __name__ == "__main__":
    # Supongamos 3 documentos de test y 4 diputados posibles
    # Documento 0: Es del Diputado índice 2
    # Documento 1: Es del Diputado índice 0
    # Documento 2: Es del Diputado índice 3
    true_mps = [2, 0, 3] 
    
    # Scores que dio el modelo (filas=docs, cols=mps)
    # Doc 0: El modelo cree que es el MP 2 (score 0.9). Acierto R@1.
    # Doc 1: El modelo cree que es MP 1 (0.8), luego MP 0 (0.6). Acierto R@2.
    # Doc 2: El modelo falla mucho.
    scores = [
        [0.1, 0.2, 0.9, 0.1], 
        [0.6, 0.8, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.1]  # MP 3 tiene el score mas bajo aqui
    ]
    
    ev = IREvaluator()
    results = ev.compute_metrics(true_mps, scores)
    print(results)