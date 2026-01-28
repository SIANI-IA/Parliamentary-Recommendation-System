
import os
import pickle

from utils import FOLDER_TO_SAVE_RESULTS


class Recommender:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.folder_to_save_results = FOLDER_TO_SAVE_RESULTS

    def __str__(self):
        return f"Recommender"

    def save_artifacts(self, model_name, output_dir, y_true, scores, threshold, metrics, mp_names_ordered):
        """
        Guarda los resultados de la inferencia para análisis posterior.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results_data = {
            "model_name": model_name,
            "threshold": threshold,
            "metrics": metrics,
            "mp_names": mp_names_ordered,       # ¡Muy importante!
            "y_true": y_true,                   # Ground truth
            "scores": scores                    # Matriz de predicciones
        }
        
        # Guardamos como .pkl
        file_path = os.path.join(output_dir, f"{model_name}_results.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(results_data, f)
            
        print(f"Resultados guardados en: {file_path}")

    def run_evaluation(self):
        raise NotImplementedError("Subclasses should implement this method.")