import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Descargar stop words si no las tienes
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ParliamentaryVectorization:
    def __init__(self, subset_path):
        self.subset_path = subset_path
        self.train_path = os.path.join(subset_path, "train.json")
        self.vectorizer = None
        self.tfidf_matrix = None
        # Mapeo para saber qué fila de la matriz corresponde a qué diputado
        self.mp_indices = {} 
        self.all_documents = []
        
    def load_and_vectorize(self):
        print(f"--- Cargando datos de: {self.train_path} ---")
        
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"No se encontró {self.train_path}")

        with open(self.train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Aplanar los datos para crear el Corpus Global
        # Necesitamos una lista única de textos para entrenar el TF-IDF
        current_idx = 0
        
        # Stop words en español
        spanish_stop_words = stopwords.words('spanish')

        # Configuración del Vectorizer (similar a lo estándar en papers de IR)
        # min_df=5: Ignora palabras que aparecen en menos de 5 documentos (quita ruido/typos)
        self.vectorizer = TfidfVectorizer(
            stop_words=spanish_stop_words,
            min_df=5, 
            sublinear_tf=True # Aplica 1+log(tf), suaviza frecuencias muy altas
        )

        print("-> Preparando corpus...")
        for mp, interventions in data.items():
            # Guardamos los índices de inicio y fin para este MP
            start = current_idx
            self.all_documents.extend(interventions)
            end = current_idx + len(interventions)
            
            # Guardamos qué filas de la matriz pertenecen a este MP
            self.mp_indices[mp] = list(range(start, end))
            current_idx = end

        print(f"-> Entrenando TF-IDF en {len(self.all_documents)} documentos...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_documents)
        
        print(f"   [OK] Matriz generada. Dimensiones: {self.tfidf_matrix.shape}")
        print(f"   (Documentos: {self.tfidf_matrix.shape[0]}, Vocabulario: {self.tfidf_matrix.shape[1]})")

    def get_data_for_mp(self, target_mp):
        """
        Retorna las matrices listas para el algoritmo PUL de un diputado específico.
        Positive (P): Sus intervenciones.
        Unlabeled (U): Las intervenciones de TODOS los demás.
        """
        if target_mp not in self.mp_indices:
            raise ValueError(f"Diputado {target_mp} no encontrado.")

        # Índices de los positivos
        pos_indices = self.mp_indices[target_mp]
        
        # Índices de los unlabeled (todos los índices que NO son del target_mp)
        all_indices = set(range(self.tfidf_matrix.shape[0]))
        unlabeled_indices = list(all_indices - set(pos_indices))
        
        # Extraer sub-matrices (slicing es eficiente en matrices dispersas CSR)
        P_matrix = self.tfidf_matrix[pos_indices]
        U_matrix = self.tfidf_matrix[unlabeled_indices]
        
        return P_matrix, U_matrix

# --- PRUEBA DEL CÓDIGO ---
if __name__ == "__main__":
    # Probamos con el subset '25' por ejemplo
    subset = "dataset/parcanDeb-mp/25"
    
    if os.path.exists(subset):
        # Inicializar
        pv = ParliamentaryVectorization(subset)
        
        # Ejecutar vectorización
        pv.load_and_vectorize()
        
        # Probar extraccion para un diputado aleatorio
        sample_mp = list(pv.mp_indices.keys())[0]
        P, U = pv.get_data_for_mp(sample_mp)
        
        print(f"\nEjemplo para diputado: {sample_mp}")
        print(f"Dimensiones Positivos (sus intervenciones): {P.shape}")
        print(f"Dimensiones Unlabeled (resto del parlamento): {U.shape}")
        print("\nListo para pasar al algoritmo PUL-KM.")
    else:
        print("No se encuentra la ruta del dataset. Ejecuta el script anterior primero.")