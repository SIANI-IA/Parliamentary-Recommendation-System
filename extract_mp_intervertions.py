import re
import json
import os
import csv
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Any

# --- CONFIGURACIÓN ---
FOLDER_DS = "/home/miguel/data/raw/CAN_dataset"
OUTPUT_BASE_DIR = "dataset/parcanDeb-mp"
MAX_CHAR_NAME = 40       # Límite caracteres nombre diputado
MIN_CHAR_CONTENT = 300    # <--- NUEVO FILTRO: Mínimo de caracteres para guardar la intervención
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Umbrales de filtrado (cantidad mínima de intervenciones por diputado)
THRESHOLDS = {
    'all': 0,
    '10': 10,
    '25': 25,
    '75': 75,
    '150': 150
}

def extract_interventions(texto: str) -> Dict[str, list]:
    """
    Extrae las intervenciones de un texto plano usando Regex y filtra las muy cortas.
    """
    intervenciones = {}
    
    # Patrón: (El señor|La señora) + (Nombre) + :
    patron = r'(El señor|La señora)\s+([^:]+):'
    matches = list(re.finditer(patron, texto))

    for i, match in enumerate(matches):
        nombre_crudo = match.group(2).strip()
        
        # --- Limpieza del nombre ---
        nombre_limpio = nombre_crudo.replace("(desde su escaño)", "").strip()
        nombre_temp = nombre_limpio.replace("(Desde su escaño)", "").strip()

        # Priorizar nombre dentro de paréntesis si existe: Ej: (Clavijo Batlle)
        match_nombre_real = re.search(r'\((.*?)\)', nombre_temp)
        
        if match_nombre_real:
            nombre_limpio = match_nombre_real.group(1).strip()
        else:
            nombre_limpio = nombre_temp

        nombre_limpio = nombre_limpio.title() # Capitalizar

        # --- Extracción del texto ---
        inicio_texto = match.end()
        if i + 1 < len(matches):
            fin_texto = matches[i+1].start()
        else:
            fin_texto = len(texto)

        contenido = texto[inicio_texto:fin_texto].strip()

        # --- FILTRADO Y GUARDADO ---
        # Solo guardamos si supera el mínimo de caracteres (quita "Sí.", "Gracias.", etc.)
        if len(contenido) >= MIN_CHAR_CONTENT:
            if nombre_limpio not in intervenciones:
                intervenciones[nombre_limpio] = []
            
            intervenciones[nombre_limpio].append(contenido)

    return intervenciones

def calculate_statistics(mp_data: Dict[str, List[str]], tokenizer) -> Dict[str, Any]:
    """
    Calcula estadísticas detalladas del subconjunto de datos.
    """
    print("  -> Calculando estadísticas y tokenizando...")
    
    total_mps = len(mp_data)
    if total_mps == 0:
        return {}

    counts_interventions = [len(msgs) for msgs in mp_data.values()]
    total_interventions = sum(counts_interventions)
    
    # Tokenización para estadísticas de longitud
    all_token_lengths = []
    # Usamos tqdm para ver progreso si son muchos datos
    for msgs in tqdm(mp_data.values(), desc="    Tokenizando intervenciones"):
        for msg in msgs:
            tokens = tokenizer.tokenize(msg)
            all_token_lengths.append(len(tokens))
    
    stats = {
        "Total MPs": total_mps,
        "Total Interventions": total_interventions,
        "Min Interventions per MP": min(counts_interventions) if counts_interventions else 0,
        "Max Interventions per MP": max(counts_interventions) if counts_interventions else 0,
        "Avg Interventions per MP": round(np.mean(counts_interventions), 2) if counts_interventions else 0,
        "Median Interventions per MP": round(np.median(counts_interventions), 2) if counts_interventions else 0,
        "Min Length (tokens)": min(all_token_lengths) if all_token_lengths else 0,
        "Max Length (tokens)": max(all_token_lengths) if all_token_lengths else 0,
        "Avg Length (tokens)": round(np.mean(all_token_lengths), 2) if all_token_lengths else 0,
        "Total Tokens in Corpus": sum(all_token_lengths)
    }
    
    return stats

def save_subset(subset_name: str, data: Dict[str, List[str]], stats: Dict[str, Any]):
    """
    Guarda el JSON de intervenciones y el CSV de estadísticas.
    """
    folder_path = os.path.join(OUTPUT_BASE_DIR, subset_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # 1. Guardar JSON
    json_path = os.path.join(folder_path, "mp_interventions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 2. Guardar CSV de estadísticas
    if stats:
        csv_path = os.path.join(folder_path, "stats_summary.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in stats.items():
                writer.writerow([key, value])
    
    print(f"  [OK] Guardado subset '{subset_name}' en: {folder_path}")

def main():
    # 1. Carga inicial
    print("Cargando dataset y tokenizador...")
    try:
        ds = load_from_disk(FOLDER_DS)['all']
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. Procesamiento Global
    master_interventions = {}
    
    print(f"Procesando transcripciones (Filtrando < {MIN_CHAR_CONTENT} caracteres)...")
    for item in tqdm(ds, desc="Extracting"):
        transcripts = item['text']
        partial_result = extract_interventions(transcripts)
        
        for mp, msgs in partial_result.items():
            # Filtros de nombre
            if mp in ["Presidenta", "Presidente"]:
                continue
            if len(mp) > MAX_CHAR_NAME:
                continue
            
            if mp not in master_interventions:
                master_interventions[mp] = []
            master_interventions[mp].extend(msgs)

    # Limpieza final: Eliminar MPs que se quedaron sin intervenciones tras el filtro de longitud
    master_interventions = {k: v for k, v in master_interventions.items() if v}

    print(f"\nExtracción base completada. Total MPs con intervenciones válidas: {len(master_interventions)}")

    # 3. Generación de Subconjuntos
    for name, threshold in THRESHOLDS.items():
        print(f"\n--- Generando subset: {name} (Min Intervenciones: {threshold}) ---")
        
        # Filtrar diccionario por número de intervenciones acumuladas
        if threshold == 0:
            subset_data = master_interventions
        else:
            subset_data = {
                k: v for k, v in master_interventions.items() 
                if len(v) >= threshold
            }
        
        if not subset_data:
            print(f"  [!] Alerta: El subset '{name}' está vacío. Se omitirá.")
            continue

        # Calcular estadísticas
        subset_stats = calculate_statistics(subset_data, tokenizer)
        
        print(f"  MPs: {subset_stats['Total MPs']} | Intervenciones: {subset_stats['Total Interventions']}")
        
        # Guardar
        save_subset(name, subset_data, subset_stats)

if __name__ == "__main__":
    main()
