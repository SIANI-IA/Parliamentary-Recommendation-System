import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from skmultilearn.model_selection import IterativeStratification

from utils import SEED  

import random

# Fijar semilla globalmente
random.seed(SEED)
np.random.seed(SEED)

# --- CONFIGURACIÓN ---
INPUT_DIR = "dataset/parcanDeb-rec"
OUTPUT_DIR = "dataset/parcanDeb-rec-split"
TINY_SIZE = 50

def main():
    print(f"Cargando dataset desde {INPUT_DIR}...")
    ds = load_from_disk(INPUT_DIR)

    # Cargar Metadata
    with open(os.path.join(INPUT_DIR, "mp_mapping.json"), 'r') as f:
        mapping_data = json.load(f)
        num_classes = len(mapping_data['label2id'])

    # Convertir a Pandas y extraer matriz de etiquetas (y)
    df = ds.to_pandas()
    # Convertir la lista de listas 'label' a una matriz NumPy (N_samples x N_classes)
    y_all = np.array(df['label'].tolist())
    
    # Índices originales para poder recuperar las filas luego
    indices_all = np.arange(len(df)).reshape(-1, 1)

    print(f"Dataset total: {y_all.shape[0]} filas, {y_all.shape[1]} etiquetas (MPs).")

    # ---------------------------------------------------------
    # PASO 1: PROTECCIÓN DE SINGLETONS Y CASOS MUY RAROS
    # ---------------------------------------------------------
    # Contamos cuántas veces aparece cada etiqueta (columna)
    label_counts = y_all.sum(axis=0)
    
    # Identificar etiquetas que aparecen solo 1 vez (Singletons)
    singleton_indices = np.where(label_counts == 1)[0]
    
    # Identificar las filas que contienen esos singletons
    # Si una fila tiene un 1 en alguna columna 'singleton', esa fila ES singleton.
    rows_with_singletons = np.any(y_all[:, singleton_indices] == 1, axis=1)
    
    # Separamos los datos en dos grupos:
    # A) Force Train: Filas con oradores únicos
    # B) Stratifiable: El resto
    idx_force_train = indices_all[rows_with_singletons]
    idx_stratifiable = indices_all[~rows_with_singletons]
    
    y_stratifiable = y_all[~rows_with_singletons]
    
    print(f" -> Filas forzadas a Train (Singletons): {len(idx_force_train)}")
    print(f" -> Filas para estratificación iterativa: {len(idx_stratifiable)}")

    # ---------------------------------------------------------
    # PASO 2: ESTRATIFICACIÓN ITERATIVA (80% Train, 20% Temp)
    # ---------------------------------------------------------
    # IterativeStratification necesita dividir X e y. Usamos los índices como X.
    k_fold = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])
    
    # Este iterador devuelve índices de train y test. Solo necesitamos la primera vuelta.
    # Nota: skmultilearn devuelve train_index, test_index. 
    # Aquí 'test_index' será nuestro 20% (Temp) y 'train_index' el 80% (Train 1)
    train_idx_iter, temp_idx_iter = next(k_fold.split(idx_stratifiable, y_stratifiable))

    # Mapeamos los índices relativos de vuelta a los índices originales del dataframe
    real_train_idx = idx_stratifiable[train_idx_iter].flatten()
    real_temp_idx = idx_stratifiable[temp_idx_iter].flatten()

    # ---------------------------------------------------------
    # PASO 3: DIVIDIR TEMP EN DEV (50%) Y TEST (50%)
    # ---------------------------------------------------------
    # Obtenemos las etiquetas del subconjunto temporal para volver a estratificar
    y_temp = y_all[real_temp_idx]
    
    # Usamos Iterative de nuevo para dividir el 20% restante en 10% Dev y 10% Test
    k_fold_2 = IterativeStratification(n_splits=2, order=1)
    dev_rel, test_rel = next(k_fold_2.split(real_temp_idx, y_temp))
    
    real_dev_idx = real_temp_idx[dev_rel].flatten()
    real_test_idx = real_temp_idx[test_rel].flatten()

    # ---------------------------------------------------------
    # PASO 4: RECONSTRUCCIÓN Y SEGURIDAD FINAL
    # ---------------------------------------------------------
    # Train Final = (Forzados) + (Estratificados)
    final_train_indices = np.concatenate([idx_force_train.flatten(), real_train_idx])
    
    # Crear Dataframes
    df_train = df.iloc[final_train_indices]
    df_dev = df.iloc[real_dev_idx]
    df_test = df.iloc[real_test_idx]

    # Convertir a Dataset HF
    train_ds = Dataset.from_pandas(df_train, preserve_index=False)
    dev_ds = Dataset.from_pandas(df_dev, preserve_index=False)
    test_ds = Dataset.from_pandas(df_test, preserve_index=False)

    print(f"\n[Splits Generados]")
    print(f" Train: {len(train_ds)}")
    print(f" Dev:   {len(dev_ds)}")
    print(f" Test:  {len(test_ds)}")

    # ---------------------------------------------------------
    # PASO 5: VALIDACIÓN DE "CLOSED WORLD"
    # ---------------------------------------------------------
    print("\nValidando consistencia de oradores...")
    
    # Obtener oradores que existen en train
    train_speakers_set = set()
    for s_list in train_ds['Speakers']:
        train_speakers_set.update(s_list)
    
    def sanitize(example):
        """Elimina oradores en dev/test que no estén en train"""
        current = example['Speakers']
        valid = [s for s in current if s in train_speakers_set]
        
        # Si la lista cambia, necesitamos actualizar el vector label
        if len(valid) != len(current):
            new_label = np.zeros(num_classes, dtype=int)
            for s in valid:
                new_label[mapping_data['label2id'][s]] = 1
            return {"Speakers": valid, "label": new_label.tolist(), "keep": len(valid) > 0}
        
        return {"Speakers": current, "label": example['label'], "keep": True}

    # Aplicar sanitización (por si IterativeStratification dejó algún caso borde raro)
    dev_ds = dev_ds.map(sanitize).filter(lambda x: x['keep']).remove_columns(['keep'])
    test_ds = test_ds.map(sanitize).filter(lambda x: x['keep']).remove_columns(['keep'])

    print(f" Train Final: {len(train_ds)} (Oradores únicos: {len(train_speakers_set)})")
    print(f" Dev Final:   {len(dev_ds)}")
    print(f" Test Final:  {len(test_ds)}")

    # 6. Guardar y Tiny
    tiny_ds = train_ds.shuffle(seed=SEED).select(range(min(TINY_SIZE, len(train_ds))))
    
    final_splits = DatasetDict({
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds,
        'tiny': tiny_ds
    })

    final_splits.save_to_disk(OUTPUT_DIR)
    
    # Copiar JSON
    with open(os.path.join(OUTPUT_DIR, "mp_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Guardado en {OUTPUT_DIR}")

if __name__ == "__main__":
    main()