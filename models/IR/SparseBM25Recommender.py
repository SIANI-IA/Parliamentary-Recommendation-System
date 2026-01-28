import numpy as np
import os
import json
from datasets import load_from_disk
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk

# LangChain imports
from langchain_community.retrievers import BM25Retriever

# Imports de tu proyecto
from eval.Evaluator import Evaluator
from models.Recommender import Recommender

class SparseBM25Recommender(Recommender):
    def __init__(self, dataset_path: str, lang: str = "spanish"):
        super().__init__(dataset_path)
        self.retriever = None
        self.mp_names_ordered = []  # Para saber quién es la fila i
        self.lang = lang
        self.stop_words = set()
        
        # Cargar mapeo para asegurar el orden correcto de los IDs
        with open(os.path.join(dataset_path, "mp_mapping.json"), 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.num_mps = len(self.id2label)

        # Configurar stopwords igual que en la clase TF-IDF
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words(self.lang))

    def _preprocess_func(self, text):
        """
        Función de preprocesamiento custom para pasarle al BM25 de LangChain.
        Replica la tokenización simple y stopwords.
        """
        # Tokenización simple por espacios y lowercase
        tokens = text.lower().split()
        return [t for t in tokens if t not in self.stop_words]

    def _create_profiles(self):
        """
        Crea un 'Macro-Documento' por diputado concatenando todo su TRAIN.
        Inicializa el BM25Retriever de LangChain.
        """
        print("1. Construyendo Perfiles de Diputados (Estrategia ir-p)...")
        train_ds = load_from_disk(self.dataset_path)['train']
        
        # Diccionario acumulador: {mp_id: [lista_textos]}
        mp_texts = {i: [] for i in range(self.num_mps)}
        
        for row in tqdm(train_ds, desc="Agrupando textos"):
            speakers = row['Speakers']
            interventions = row['Interventions']
            
            label2id = {v: k for k, v in self.id2label.items()}
            
            for sp, texts in zip(speakers, interventions):
                if sp in label2id:
                    mid = label2id[sp]
                    mp_texts[mid].extend(texts)
        
        # Crear corpus ordenado por ID
        corpus_profiles = []
        for i in range(self.num_mps):
            full_text = " ".join(mp_texts[i])
            corpus_profiles.append(full_text)
            self.mp_names_ordered.append(self.id2label[i])

        print("2. Indexando con LangChain BM25Retriever...")
        
        # Instanciamos el Retriever. 
        # Usamos preprocess_func para ser justos en la comparación con TF-IDF (stopwords).
        self.retriever = BM25Retriever.from_texts(
            corpus_profiles, 
            preprocess_func=self._preprocess_func
        )
        
        # Establecemos k igual al total de MPs para poder recuperar scores de todos si quisiéramos
        self.retriever.k = self.num_mps
        
        print(f"   [OK] Índice BM25 creado con {self.num_mps} perfiles.")

    def _compute_scores_matrix(self, text_list):
        """
        Calcula la matriz de similitud (N_querys, N_MPs) usando BM25.
        NOTA: Accedemos al objeto interno 'vectorizer' (rank_bm25) para obtener
        la matriz densa de scores, ya que el método estándar 'invoke' solo devuelve docs.
        """
        scores_matrix = []
        
        # El objeto interno rank_bm25
        bm25_obj = self.retriever.vectorizer
        
        for text in tqdm(text_list, desc="Calculando BM25 Scores"):
            # Tokenizamos la query igual que los documentos
            tokenized_query = self._preprocess_func(text)
            # get_scores devuelve un array de scores para todos los documentos del corpus
            doc_scores = bm25_obj.get_scores(tokenized_query)
            scores_matrix.append(doc_scores)
            
        scores_matrix = np.array(scores_matrix)
        
        # --- NORMALIZACIÓN ---
        # BM25 no devuelve valores entre 0 y 1 (puede ser 0 a 40+).
        # Para que el threshold optimizer y el Evaluator funcionen bien, normalizamos.
        # Normalizamos por fila (por query) o globalmente. 
        # Aquí usaremos MinMax global o por query. Por consistencia con cosenos, 
        # dividiremos por el score máximo observado en la matriz para mantener la escala relativa.
        
        max_val = np.max(scores_matrix)
        if max_val > 0:
            scores_matrix = scores_matrix / max_val
            
        return scores_matrix

    def _optimize_threshold(self, y_true, scores):
        """Busca el mejor umbral en DEV para maximizar Micro-F1"""
        print("\n=== Optimizando Umbral en DEV (BM25) ===")
        # Como hemos normalizado los scores a [0, 1], podemos usar el mismo linspace
        thresholds = np.linspace(0.01, 0.99, 50)
        best_f1 = -1.0
        best_th = 0.05
        
        for th in thresholds:
            y_pred_bin = (scores >= th).astype(int)
            
            tp = np.sum((y_pred_bin == 1) & (y_true == 1))
            fp = np.sum((y_pred_bin == 1) & (y_true == 0))
            fn = np.sum((y_pred_bin == 0) & (y_true == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        print(f" -> Mejor Umbral BM25: {best_th:.4f} (F1: {best_f1:.4f})")
        return best_th

    def run_evaluation(self):
        # 1. Train (Indexar)
        self._create_profiles()
        
        # 2. Validación (Dev)
        print("\n3. Validando en DEV...")
        dev_ds = load_from_disk(self.dataset_path)['dev']
        y_dev_true = np.array(dev_ds['label'])
        
        # Calcular scores BM25
        scores_dev = self._compute_scores_matrix(dev_ds['Text'])
        
        best_threshold = self._optimize_threshold(y_dev_true, scores_dev)
        
        # 3. Test
        print("\n4. Evaluando en TEST...")
        test_ds = load_from_disk(self.dataset_path)['test']
        y_test_true = np.array(test_ds['label'])
        
        scores_test = self._compute_scores_matrix(test_ds['Text'])
        
        # Evaluación
        evaluator = Evaluator(k_values=[1, 5, 10, 20])
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