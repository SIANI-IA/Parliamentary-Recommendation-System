import numpy as np

SEED = 123
FOLDER_TO_SAVE_RESULTS = "results"

def estimate_prior(dataset):
    """
    Estima pi contando cuántos 1s hay en total en el dataset de entrenamiento
    dividido por el número total de celdas (docs * mps).
    """
    total_ones = 0
    total_elements = 0
    
    # Iteramos rápido (asumiendo formato 'labels' como tensores o listas)
    # Si dataset es muy grande, hazlo sobre una muestra representativa
    for row in dataset:
        # row['labels'] es un vector de 0s y 1s
        labels = np.array(row['labels']) 
        total_ones += np.sum(labels)
        total_elements += len(labels)
        
    prior = total_ones / total_elements
    return prior