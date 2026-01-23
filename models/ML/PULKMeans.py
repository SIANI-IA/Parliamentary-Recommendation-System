import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

class PULKMeans:
    def __init__(self, max_iter: int = 20, tol: float = 1e-4, verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.reliable_negatives_indices = []

    def fit(self, P_matrix, U_matrix):
        """
        P_matrix: Sparse matrix con las intervenciones del diputado (Positivos)
        U_matrix: Sparse matrix con el resto de intervenciones (Unlabeled)
        """
        n_pos = P_matrix.shape[0]
        n_unlabeled = U_matrix.shape[0]
        
        # --- 1. INICIALIZACIÓN (Section 3.1) ---
        # "positive centroid is computed from all the positive examples" [cite: 93]
        # "negative centroid is calculated from all the unlabeled examples" [cite: 93]
        # Nota: .mean en sparse matrix devuelve np.matrix, lo convertimos a np.array
        c_pos = np.array(P_matrix.mean(axis=0))
        c_neg = np.array(U_matrix.mean(axis=0))
        
        # Etiquetas iniciales para U (0 = Negativo, 1 = Positivo)
        # Al principio todos los Unlabeled contribuyen al centroide negativo
        u_labels = np.zeros(n_unlabeled) 
        
        if self.verbose:
            print(f"  [PUL-KM] Inicio: {n_pos} Positivos vs {n_unlabeled} Unlabeled")

        for iteration in range(self.max_iter):
            prev_u_labels = u_labels.copy()

            # --- 2. ASIGNACIÓN (The proposed modification) ---
            # Calculamos similitud coseno de U contra los dos centroides [cite: 91]
            # U vs Centroid Positive
            sim_u_p = cosine_similarity(U_matrix, c_pos).flatten()
            # U vs Centroid Negative
            sim_u_n = cosine_similarity(U_matrix, c_neg).flatten()

            # Asignamos cada documento de U al cluster más similar (mayor coseno)
            # "unlabeled examples can fluctuate between the two clusters" [cite: 92]
            u_labels = (sim_u_p > sim_u_n).astype(int)

            # --- 3. ACTUALIZACIÓN DE CENTROIDES ---
            # Cluster Positivo Actual: TODOS los P originales + los U que se parecen a P
            # "known positive examples are forced to always remain in the positive cluster" [cite: 92]
            
            # Identificar qué documentos de U se movieron al cluster positivo
            u_assigned_to_pos = U_matrix[u_labels == 1]
            
            # Si hay nuevos miembros en el cluster positivo, actualizamos c_pos
            if u_assigned_to_pos.shape[0] > 0:
                # Concatenamos P original + U "positivos"
                current_cluster_pos = vstack([P_matrix, u_assigned_to_pos])
                c_pos = np.array(current_cluster_pos.mean(axis=0))
            else:
                # Si ningún U se parece a P, el centroide es solo P (como al inicio)
                c_pos = np.array(P_matrix.mean(axis=0))

            # Cluster Negativo Actual: Solo los U que se quedaron en el cluster negativo
            u_assigned_to_neg = U_matrix[u_labels == 0]
            
            if u_assigned_to_neg.shape[0] > 0:
                c_neg = np.array(u_assigned_to_neg.mean(axis=0))
            else:
                # Caso extremo (raro): Todos los U se fueron a P. 
                # No actualizamos c_neg o detenemos.
                if self.verbose:
                    print("  [Warning] Cluster negativo vacío.")
                break

            # --- 4. CONVERGENCIA ---
            # Si las etiquetas de U no cambian, hemos convergido [cite: 90]
            changes = np.sum(u_labels != prev_u_labels)
            if changes == 0:
                if self.verbose:
                    print(f"  [PUL-KM] Convergencia alcanzada en iteración {iteration+1}")
                break
        
        # --- 5. RESULTADO FINAL ---
        # "unlabeled examples which remain in the negative cluster are considered reliable negative" [cite: 94]
        # Guardamos los índices RELATIVOS dentro de U_matrix
        self.reliable_negatives_indices = np.where(u_labels == 0)[0]
        
        num_rn = len(self.reliable_negatives_indices)
        if self.verbose:
            print(f"  [PUL-KM] Fin. Detectados {num_rn} Negativos Fiables (de {n_unlabeled} Unlabeled)")
        
        return self.reliable_negatives_indices