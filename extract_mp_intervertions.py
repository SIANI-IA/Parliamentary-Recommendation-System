import re
import json
import os
import numpy as np
import csv
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Any

# --- CONFIGURACIÓN ---
FOLDER_DS = "/home/miguel/data/raw/CAN_dataset"
OUTPUT_DIR = "dataset/parcanDeb-rec"
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_CHAR_NAME = 40       # Límite caracteres nombre diputado
MIN_CHAR_CONTENT = 300   # Filtro: Mínimo de caracteres para guardar la intervención

def extract_interventions(texto: str) -> Dict[str, list]:
    """
    Extrae las intervenciones de un texto plano usando Regex.
    Devuelve: { 'NombreMP': ['texto1', 'texto2'] }
    """
    intervenciones = {}
    patron = r'(El señor|La señora)\s+([^:]+):'
    matches = list(re.finditer(patron, texto))

    for i, match in enumerate(matches):
        nombre_crudo = match.group(2).strip()
        
        # --- Limpieza del nombre ---
        nombre_limpio = nombre_crudo.replace("(desde su escaño)", "").strip()
        nombre_temp = nombre_limpio.replace("(Desde su escaño)", "").strip()

        match_nombre_real = re.search(r'\((.*?)\)', nombre_temp)
        if match_nombre_real:
            nombre_limpio = match_nombre_real.group(1).strip()
        else:
            nombre_limpio = nombre_temp

        nombre_limpio = nombre_limpio.title()

        # --- Extracción del texto ---
        inicio_texto = match.end()
        if i + 1 < len(matches):
            fin_texto = matches[i+1].start()
        else:
            fin_texto = len(texto)

        contenido = texto[inicio_texto:fin_texto].strip()

        # --- FILTRADO POR LONGITUD ---
        if len(contenido) >= MIN_CHAR_CONTENT:
            if nombre_limpio not in intervenciones:
                intervenciones[nombre_limpio] = []
            intervenciones[nombre_limpio].append(contenido)

    return intervenciones

def save_statistics(output_path: str, stats: Dict[str, Any]):
    """Guarda las estadísticas en JSON y CSV para fácil lectura"""
    # Guardar JSON
    with open(os.path.join(output_path, "stats_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    # Guardar CSV simple
    with open(os.path.join(output_path, "stats_summary.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            writer.writerow([k, v])

def main():
    # 1. Carga inicial
    print(f"Cargando dataset desde {FOLDER_DS}...")
    try:
        ds = load_from_disk(FOLDER_DS)['all']
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    print(f"Cargando tokenizador ({MODEL_ID}) para estadísticas...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except:
        print(" [!] No se pudo cargar el tokenizador. Usando conteo de palabras simple como fallback.")
        tokenizer = None

    processed_rows = []
    
    # Estructuras para estadísticas
    stats_mp_counts = {}      # { "Diputado A": 5, "Diputado B": 10 }
    stats_token_lengths = []  # [120, 500, 45, ...] longitud de cada intervención
    
    print("Procesando filas, extrayendo oradores y calculando estadísticas...")
    
    # 2. Primera Pasada: Extracción, Descubrimiento y Stats
    for item in tqdm(ds, desc="Processing"):
        original_text = item['text']
        original_pk = item.get('pk', item.get('PK', 'Unknown'))
        
        interventions_dict = extract_interventions(original_text)
        
        row_speakers = []
        row_interventions = []

        for mp, texts in interventions_dict.items():
            # Filtros de limpieza de nombres
            if mp in ["Presidenta", "Presidente"] or len(mp) > MAX_CHAR_NAME:
                continue
            
            # --- Estadísticas ---
            # 1. Conteo de intervenciones por MP
            stats_mp_counts[mp] = stats_mp_counts.get(mp, 0) + len(texts)
            
            # 2. Longitud de tokens por intervención
            for text in texts:
                if tokenizer:
                    # Tokenización real (más lento pero exacto para LLMs)
                    length = len(tokenizer.tokenize(text))
                else:
                    # Fallback aproximado (palabras)
                    length = len(text.split())
                stats_token_lengths.append(length)

            # --- Guardado de fila ---
            row_speakers.append(mp)
            row_interventions.append(texts)
        
        # Añadir fila si tiene oradores válidos
        if len(row_speakers) > 0:
            processed_rows.append({
                'PK': original_pk,
                'Text': original_text,
                'Speakers': row_speakers,
                'Interventions': row_interventions
            })

    # 3. Calcular Métricas Finales
    total_mps_unique = len(stats_mp_counts)
    counts_values = list(stats_mp_counts.values())
    
    statistics = {
        "Total Documents (Initiatives)": len(processed_rows),
        "Total Unique MPs": total_mps_unique,
        "Total Interventions Extracted": sum(counts_values) if counts_values else 0,
        "Total Tokens in Corpus": sum(stats_token_lengths),
        
        # Stats por Diputado (Balanceo de clases)
        "Interventions per MP (Min)": min(counts_values) if counts_values else 0,
        "Interventions per MP (Max)": max(counts_values) if counts_values else 0,
        "Interventions per MP (Avg)": round(np.mean(counts_values), 2) if counts_values else 0,
        "Interventions per MP (Median)": round(np.median(counts_values), 2) if counts_values else 0,
        
        # Stats por Intervención (Context Window)
        "Tokens per Intervention (Min)": min(stats_token_lengths) if stats_token_lengths else 0,
        "Tokens per Intervention (Max)": max(stats_token_lengths) if stats_token_lengths else 0,
        "Tokens per Intervention (Avg)": round(np.mean(stats_token_lengths), 2) if stats_token_lengths else 0,
    }

    # 4. Crear Mapeo y Vectores Label
    sorted_mps = sorted(list(stats_mp_counts.keys()))
    mp_to_index = {mp: idx for idx, mp in enumerate(sorted_mps)}
    num_classes = len(sorted_mps)
    
    print(f"\nUniverso de MPs: {num_classes}. Generando columna 'label' vectorizada...")

    # 5. Segunda Pasada: Vectorización
    final_data = []
    for row in tqdm(processed_rows, desc="Vectorizing"):
        label_vector = np.zeros(num_classes, dtype=int)
        for speaker in row['Speakers']:
            if speaker in mp_to_index:
                label_vector[mp_to_index[speaker]] = 1
        
        row['label'] = label_vector.tolist()
        final_data.append(row)

    # 6. Guardado Final
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dataset HF
    print(f"Guardando dataset HF en {OUTPUT_DIR}...")
    hf_dataset = Dataset.from_list(final_data)
    hf_dataset.save_to_disk(OUTPUT_DIR)
    
    # Mapeo MP -> ID
    mapping_path = os.path.join(OUTPUT_DIR, "mp_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "id2label": {i: mp for i, mp in enumerate(sorted_mps)},
            "label2id": mp_to_index
        }, f, ensure_ascii=False, indent=4)

    # Estadísticas
    save_statistics(OUTPUT_DIR, statistics)

    print("\n[RESUMEN ESTADÍSTICO]")
    for k, v in statistics.items():
        print(f"  - {k}: {v}")

    print(f"\n[OK] Proceso completado exitosamente.")

if __name__ == "__main__":
    main()