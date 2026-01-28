import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk

# Imports de tu proyecto
from eval.Evaluator import Evaluator
from models.Recommender import Recommender

class SparseTfidfRecommender(Recommender):
    def __init__(self, dataset_path: str, lang: str = "spanish"):
        super().__init__(dataset_path)
        self.vectorizer = None
        self.profiles_matrix = None # Matriz (N_MPs, Vocabulario)
        self.mp_names_ordered = []  # Para saber quién es la fila i
        self.lang = lang
        
        # Cargar mapeo para asegurar el orden correcto de los IDs
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            mapping = json.load(f)
            # Ordenamos por ID para que la fila 0 sea el MP con ID 0, etc.
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.num_mps = len(self.id2label)

    def _create_profiles(self):
        """
        Crea un 'Macro-Documento' por diputado concatenando todo su TRAIN.
        Entrena el TF-IDF sobre estos perfiles.
        """
        print("1. Construyendo Perfiles de Diputados (Estrategia ir-p)...")
        train_ds = load_from_disk(self.dataset_path)['train']
        
        # Diccionario acumulador: {mp_id: [lista_textos]}
        mp_texts = {i: [] for i in range(self.num_mps)}
        
        for row in tqdm(train_ds, desc="Agrupando textos"):
            speakers = row['Speakers']
            interventions = row['Interventions']
            
            # Mapeamos nombre -> ID para guardar en el sitio correcto
            # Necesitamos el mapa inverso momentáneamente
            label2id = {v: k for k, v in self.id2label.items()}
            
            for sp, texts in zip(speakers, interventions):
                if sp in label2id:
                    mid = label2id[sp]
                    mp_texts[mid].extend(texts)
        
        # Crear la lista de documentos para el TF-IDF
        # El orden de esta lista DEBE ser por ID creciente (0, 1, 2...)
        corpus_profiles = []
        for i in range(self.num_mps):
            # Unimos todo el texto del diputado en un solo string
            full_text = " ".join(mp_texts[i])
            corpus_profiles.append(full_text)
            self.mp_names_ordered.append(self.id2label[i])

        # Vectorización TF-IDF
        print("2. Entrenando TF-IDF Vectorizer...")
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        stop_words = stopwords.words(self.lang)

        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            min_df=5, 
            sublinear_tf=True # 1 + log(tf) -> Importante para documentos largos concatenados
        )
        
        # Matriz (N_MPs, Vocabulario)
        self.profiles_matrix = self.vectorizer.fit_transform(corpus_profiles)
        print(f"   [OK] Perfiles vectorizados. Shape: {self.profiles_matrix.shape}")

    def _optimize_threshold(self, y_true, scores):
        """Busca el mejor umbral en DEV para maximizar Micro-F1"""
        print("\n=== Optimizando Umbral en DEV ===")
        # Los scores de Coseno van de 0 a 1.
        thresholds = np.linspace(0.01, 1, 50) # Rango típico para TF-IDF
        best_f1 = -1.0
        best_th = 0.05
        
        for th in thresholds:
            y_pred_bin = (scores >= th).astype(int)
            
            # Cálculo rápido de F1 Micro
            tp = np.sum((y_pred_bin == 1) & (y_true == 1))
            fp = np.sum((y_pred_bin == 1) & (y_true == 0))
            fn = np.sum((y_pred_bin == 0) & (y_true == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        print(f" -> Mejor Umbral: {best_th:.4f} (F1: {best_f1:.4f})")
        return best_th

    def run_evaluation(self):
        # 1. Train (Crear Índice)
        self._create_profiles()
        
        # 2. Validación (Dev)
        print("\n3. Validando en DEV...")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        X_dev = self.vectorizer.transform(dev_ds['Text'])
        y_dev_true = np.array(dev_ds['label'])
        
        # Similitud Coseno: (Docs_Dev, Vocab) x (MPs, Vocab).T -> (Docs_Dev, MPs)
        scores_dev = cosine_similarity(X_dev, self.profiles_matrix)
        
        best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 3. Test
        print("\n4. Evaluando en TEST...")
        test_ds = load_from_disk(self.dataset_path)['test']
        X_test = self.vectorizer.transform(test_ds['Text'])
        y_test_true = np.array(test_ds['label'])
        
        scores_test = cosine_similarity(X_test, self.profiles_matrix)
        
        # Evaluación
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
        # Pasamos el umbral optimizado
        metrics = evaluator.compute_all_metrics(y_test_true, scores_test, threshold=best_threshold)
        evaluator.print_report(metrics)

        dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
        folder_name = os.path.join(self.folder_to_save_results, dataset_name, self.__str__())
        self.save_artifacts(
            output_dir=folder_name, 
            y_true=y_test_true, 
            scores=scores_test, 
            threshold=best_threshold, 
            metrics=metrics
        )
        
        return metrics
    

