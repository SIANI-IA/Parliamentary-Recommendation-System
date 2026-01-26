import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_from_disk
import nltk
from nltk.corpus import stopwords

# Descargar stop words si no existen
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ParliamentaryVectorization:
    def __init__(self, dataset_path: str, lang: str = 'spanish'):
        self.dataset_path = dataset_path
        self.lang = lang
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Mapeo: mp_id -> lista de índices de filas en la matriz tfidf
        # Ejemplo: { 0: [0, 1, 10], 1: [2, 5] }
        self.mp_to_matrix_indices = {} 
        self.all_documents = [] # Lista plana de textos para vectorizar
        
        # Cargar mapeo de nombres a IDs
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            self.mapping = json.load(f)
            # Creamos mapa inverso para facilidades de debug: Nombre -> ID
            self.name_to_id = self.mapping['label2id']
            self.id_to_name = {int(k): v for k, v in self.mapping['id2label'].items()}

    def load_and_vectorize(self):
        print(f"--- Cargando TRAIN dataset desde: {self.dataset_path} ---")
        
        try:
            # Cargar solo el split de TRAIN
            ds = load_from_disk(self.dataset_path)['train']
        except Exception as e:
            raise FileNotFoundError(f"Error cargando dataset: {e}")

        print("-> Procesando intervenciones para construir el corpus...")
        
        current_idx = 0
        
        # Iterar sobre cada iniciativa
        # Estructura de row['Interventions']: [ ["texto1_mpA", "texto2_mpA"], ["texto1_mpB"] ]
        # Estructura de row['Speakers']: ["Diputado A", "Diputado B"]
        for row in ds:
            speakers = row['Speakers']
            interventions_list_of_lists = row['Interventions']
            
            # Asegurar consistencia (mismo número de speakers que listas de intervenciones)
            if len(speakers) != len(interventions_list_of_lists):
                continue
                
            for mp_name, texts in zip(speakers, interventions_list_of_lists):
                # Obtener ID numérico del MP
                if mp_name not in self.name_to_id:
                    continue # Seguridad
                mp_id = self.name_to_id[mp_name]
                
                # Inicializar lista para este MP si es nuevo
                if mp_id not in self.mp_to_matrix_indices:
                    self.mp_to_matrix_indices[mp_id] = []
                
                # Añadir textos a la lista global y guardar índices
                for text in texts:
                    self.all_documents.append(text)
                    self.mp_to_matrix_indices[mp_id].append(current_idx)
                    current_idx += 1

        print(f"-> Corpus preparado con {len(self.all_documents)} intervenciones individuales.")

        # Configuración TF-IDF
        print("-> Entrenando TfidfVectorizer...")
        spanish_stop_words = stopwords.words(self.lang)
        
        self.vectorizer = TfidfVectorizer(
            stop_words=spanish_stop_words,
            min_df=5, 
            sublinear_tf=True 
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_documents)
        
        print(f"   [OK] Matriz generada. Dimensiones: {self.tfidf_matrix.shape}")
        print(f"   (Intervenciones: {self.tfidf_matrix.shape[0]}, Vocabulario: {self.tfidf_matrix.shape[1]})")

    def get_data_for_mp(self, mp_identifier):
        """
        Retorna matrices P y U para un diputado.
        mp_identifier: Puede ser el nombre (str) o el ID (int).
        """
        # Resolver ID
        if isinstance(mp_identifier, str):
            if mp_identifier not in self.name_to_id:
                raise ValueError(f"Diputado '{mp_identifier}' no encontrado.")
            target_id = self.name_to_id[mp_identifier]
        else:
            target_id = mp_identifier

        if target_id not in self.mp_to_matrix_indices:
             raise ValueError(f"El diputado ID {target_id} no tiene intervenciones en TRAIN.")

        # Índices Positivos (P)
        pos_indices = self.mp_to_matrix_indices[target_id]
        
        # Índices Unlabeled (U) = Todos - Positivos
        # Optimización: Usar máscara booleana o sets para velocidad
        total_docs = self.tfidf_matrix.shape[0]
        pos_set = set(pos_indices)
        
        # Generar lista de unlabeled (esto puede ser grande, cuidado con memoria si son millones)
        # Para 12k documentos es instantáneo.
        unlabeled_indices = [i for i in range(total_docs) if i not in pos_set]
        
        P_matrix = self.tfidf_matrix[pos_indices]
        U_matrix = self.tfidf_matrix[unlabeled_indices]
        
        return P_matrix, U_matrix

# --- TEST ---
if __name__ == "__main__":
    DATASET_PATH = "dataset/parcanDeb-rec-split"
    
    if os.path.exists(DATASET_PATH):
        pv = ParliamentaryVectorization(DATASET_PATH)
        pv.load_and_vectorize()
        
        # Probar con un diputado real (sacamos uno del mapeo)
        test_id = list(pv.mp_to_matrix_indices.keys())[10]
        test_name = pv.id_to_name[test_id]
        
        print(f"\n--- Prueba para: {test_name} (ID: {test_id}) ---")
        P, U = pv.get_data_for_mp(test_id)
        
        print(f"Intervenciones propias (P): {P.shape[0]}")
        print(f"Resto del parlamento (U):   {U.shape[0]}")
        print(f"Total (debe coincidir):     {P.shape[0] + U.shape[0]}")
        
    else:
        print(f"No existe {DATASET_PATH}. Ejecuta el script de splitting primero.")