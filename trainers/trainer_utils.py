import random
import numpy as np
import torch
from transformers import set_seed
from sklearn.metrics import f1_score, precision_score, recall_score

def initialize_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #torch.use_deterministic_algorithms(True, warn_only=True)

def compute_metrics(p):
    predictions, labels = p
    # Convertimos logits a probabilidades
    probs = 1 / (1 + np.exp(-predictions))
    
    # Usamos un umbral estándar de 0.5 para monitorear durante el entrenamiento
    # (El umbral óptimo se calculará al final, esto es solo para ver progreso)
    y_pred = (probs > 0.5).astype(int)
    
    # Calculamos F1 Micro (la métrica más importante para ver convergencia global)
    f1 = f1_score(labels, y_pred, average='micro', zero_division=0)
    precision = precision_score(labels, y_pred, average='micro', zero_division=0)
    recall = recall_score(labels, y_pred, average='micro', zero_division=0)
    
    return {
        'f1_micro': f1,
        'precision': precision, 
        'recall': recall
    }

def estimate_prior(dataset):
    print("\n[-] Estimando el Prior del conjunto de datos...")
    total_ones = 0
    total_elements = 0
    
    # Iteramos sobre una muestra del dataset procesado (o todo si es rápido)
    for i in range(len(dataset)):
        labels = np.array(dataset[i]['label'])
        total_ones += labels.sum().item()
        total_elements += len(labels)
    
    real_prior = total_ones / total_elements
    return real_prior * 1.05

def get_pos_weight_value(type: str, num_labels: int) -> float:
    if type == 'linear':
        pos_weight_value = 10
    elif type == 'sqrt':
        pos_weight_value = np.sqrt(num_labels)
    elif type == 'log':
        pos_weight_value = np.log(num_labels)
    else:
        raise ValueError(f"Tipo de pos_weight desconocido: {type}")
    return pos_weight_value
    