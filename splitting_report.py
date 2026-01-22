import json
import os
import random
import math
import csv
from typing import Dict, List, Tuple

# --- CONFIGURACIÓN ---
BASE_DIR = "dataset/parcanDeb-mp"
SUBSETS = ['all', '10', '25', '75', '150']
SEED = 123

def split_interventions(interventions: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Divide una lista de intervenciones en Train, Val y Test siguiendo la lógica:
    - N < 3: Todo a Train (datos insuficientes para dividir).
    - 3 <= N < 10: 1 a Test, 1 a Val, resto a Train.
    - N >= 10: 10% Test, 10% Val, 80% Train (redondeo estándar).
    """
    n = len(interventions)
    
    # Aseguramos aleatoriedad determinista
    # Hacemos una copia para no modificar la lista original fuera de la función
    shuffled_data = interventions.copy()
    random.shuffle(shuffled_data)

    if n < 3:
        # Caso extremo: No hay suficiente para repartir
        return shuffled_data, [], []
    
    elif n < 10:
        # Caso pocos datos: Asegurar al menos 1 en validación y test
        test_data = shuffled_data[:1]
        val_data = shuffled_data[1:2]
        train_data = shuffled_data[2:]
        return train_data, val_data, test_data
    
    else:
        # Caso estándar: 80 / 10 / 10
        n_test = math.ceil(n * 0.10)
        n_val = math.ceil(n * 0.10)
        
        # Índices de corte
        test_end = n_test
        val_end = n_test + n_val
        
        test_data = shuffled_data[:test_end]
        val_data = shuffled_data[test_end:val_end]
        train_data = shuffled_data[val_end:]
        
        return train_data, val_data, test_data

def process_subset(subset_name: str):
    folder_path = os.path.join(BASE_DIR, subset_name)
    input_file = os.path.join(folder_path, "mp_interventions.json")

    if not os.path.exists(input_file):
        print(f"  [SKIPPING] No se encontró {input_file}")
        return None

    print(f"--- Procesando subset: {subset_name} ---")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Estructuras para los nuevos datasets
    train_set = {}
    val_set = {}
    test_set = {}

    stats = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # Iterar por cada diputado para mantener la estratificación
    for mp, interventions in data.items():
        train_i, val_i, test_i = split_interventions(interventions)

        if train_i: train_set[mp] = train_i
        if val_i: val_set[mp] = val_i
        if test_i: test_set[mp] = test_i

        stats["train"] += len(train_i)
        stats["val"] += len(val_i)
        stats["test"] += len(test_i)

    # Guardar los archivos
    splits = [("train.json", train_set), ("val.json", val_set), ("test.json", test_set)]
    
    for filename, content in splits:
        out_path = os.path.join(folder_path, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    print(f"  [OK] Guardados splits en {folder_path}")
    print(f"  Distribución -> Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    
    return stats

def main():
    random.seed(SEED) # Semilla global para reproducibilidad
    
    report_data = []

    for subset in SUBSETS:
        subset_stats = process_subset(subset)
        if subset_stats:
            total = sum(subset_stats.values())
            report_data.append({
                "Subset": subset,
                "Total Interventions": total,
                "Train Count": subset_stats["train"],
                "Val Count": subset_stats["val"],
                "Test Count": subset_stats["test"],
                "Train %": f"{subset_stats['train']/total:.1%}",
                "Val %": f"{subset_stats['val']/total:.1%}",
                "Test %": f"{subset_stats['test']/total:.1%}"
            })

    # Guardar reporte CSV general
    csv_path = os.path.join(BASE_DIR, "splitting_report.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Subset", "Total Interventions", "Train Count", "Val Count", "Test Count", "Train %", "Val %", "Test %"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_data)
    
    print(f"\n[FIN] Reporte general guardado en: {csv_path}")

if __name__ == "__main__":
    main()