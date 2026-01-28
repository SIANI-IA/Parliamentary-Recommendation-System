import json
import os
import re
import csv
import numpy as np
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from typing import Dict, Any

# --- CONFIGURACIÓN ---
INPUT_DS_PATH = "/home/miguel/data/raw/CA_dataset"  # Ruta a tu dataset original
OUTPUT_DIR = "dataset/canada-rec"            # Carpeta de salida
MIN_CHAR_CONTENT = 300                               # Mínimo caracteres para guardar intervención

# Configuración de Estadísticas
CALCULATE_TOKENS = True  # Poner en False para ir más rápido (solo cuenta palabras con split)

def clean_speaker_name(honorific, raw_name):
    """
    Limpia el nombre del orador eliminando paréntesis y cargos.
    Ej: "Hon. Ali Ehsassi (Willowdale, Lib.)" -> "Hon. Ali Ehsassi"
    """
    # 1. Eliminar contenido entre paréntesis
    name_clean = re.sub(r'\s*\(.*?\)', '', raw_name).strip()
    # 2. Reconstruir
    full_name = f"{honorific} {name_clean}".strip()
    return full_name.title()

def extract_topics_from_text(full_text: str) -> list:
    """
    Divide un texto diario en múltiples 'Topics' (iniciativas).
    Devuelve lista de dicts con estructura interna preparada.
    """
    # Regex: Honorífico + Nombre + :
    patron_orador = r'^(Hon\.|Right Hon\.|Mr\.|Ms\.|Mrs\.)\s+([^:]+)\s*:'
    
    extracted_topics = []
    
    current_topic_name = None
    current_topic_text_lines = []
    current_interventions = {} 
    current_speaker = None
    
    lines = full_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line: continue

        # --- CAMBIO DE TOPIC ---
        if line.startswith("Topic:"):
            if current_topic_name and current_interventions:
                extracted_topics.append({
                    'PK': current_topic_name,
                    'Text': "\n".join(current_topic_text_lines),
                    'interventions_map': current_interventions
                })
            
            current_topic_name = line.replace("Topic:", "").strip()
            current_topic_text_lines = [line]
            current_interventions = {}
            current_speaker = None
            continue

        if not current_topic_name: continue

        current_topic_text_lines.append(line)

        # --- DETECCIÓN ORADOR ---
        match = re.match(patron_orador, line)
        if match:
            honorific = match.group(1)
            raw_name_part = match.group(2)
            
            # Limpieza unificada
            speaker_name = clean_speaker_name(honorific, raw_name_part)
            
            contenido = line[match.end():].strip()
            current_speaker = speaker_name
            
            if current_speaker not in current_interventions:
                current_interventions[current_speaker] = []
            
            if len(contenido) > 0:
                current_interventions[current_speaker].append(contenido)
        
        # --- CONTINUACIÓN TEXTO ---
        else:
            if current_speaker:
                if not current_interventions[current_speaker]:
                    current_interventions[current_speaker].append(line)
                else:
                    current_interventions[current_speaker][-1] += " " + line

    # Guardar último bloque
    if current_topic_name and current_interventions:
        extracted_topics.append({
            'PK': current_topic_name,
            'Text': "\n".join(current_topic_text_lines),
            'interventions_map': current_interventions
        })
        
    return extracted_topics

def save_statistics(output_path: str, stats: Dict[str, Any]):
    """Guarda las estadísticas en JSON y CSV"""
    # Guardar JSON
    with open(os.path.join(output_path, "stats_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    # Guardar CSV
    with open(os.path.join(output_path, "stats_summary.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            writer.writerow([k, v])

def main():
    # 1. Carga Dataset Original
    print(f"Cargando dataset desde {INPUT_DS_PATH}...")
    try:
        ds_raw = load_from_disk(INPUT_DS_PATH)
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # 2. Extracción y Recolección de Stats
    print("Procesando textos, extrayendo topics y calculando estadísticas...")
    
    all_processed_rows = []
    
    # Acumuladores para estadísticas
    stats_mp_counts = {}      # { "Hon. John Nater": 50 intervenciones }
    stats_token_lengths = []  # [120, 45, 10, ...] palabras por intervención
    
    for row in tqdm(ds_raw, desc="Processing"):
        original_text = row['text'] # Asumo que la columna se llama 'text' o 'Text'
        
        # Extraer topics de esta sesión
        topics_found = extract_topics_from_text(original_text)
        
        for topic in topics_found:
            # Procesar intervenciones dentro del topic
            valid_interventions_map = {}
            has_content = False
            
            for mp, texts in topic['interventions_map'].items():
                # Filtro longitud mínima
                filtered_texts = [t for t in texts if len(t) >= MIN_CHAR_CONTENT]
                
                if filtered_texts:
                    valid_interventions_map[mp] = filtered_texts
                    has_content = True
                    
                    # --- STATS: Conteo por MP ---
                    stats_mp_counts[mp] = stats_mp_counts.get(mp, 0) + len(filtered_texts)
                    
                    # --- STATS: Longitud (aproximada por espacios) ---
                    for t in filtered_texts:
                        # Usamos split() como proxy de tokens. 
                        # Si quieres exactitud LLM, usa len(tokenizer.encode(t))
                        length = len(t.split())
                        stats_token_lengths.append(length)

            # Solo añadimos el topic si tiene intervenciones válidas
            if has_content:
                topic['interventions_map'] = valid_interventions_map
                all_processed_rows.append(topic)

    # 3. Calcular Métricas Finales
    counts_values = list(stats_mp_counts.values())
    total_mps = len(stats_mp_counts)
    
    statistics = {
        "Total Documents (Topics/Initiatives)": len(all_processed_rows),
        "Total Unique MPs": total_mps,
        "Total Interventions Extracted": sum(counts_values) if counts_values else 0,
        "Total Tokens (approx words)": sum(stats_token_lengths),
        
        # Distribución por Diputado
        "Interventions per MP (Min)": min(counts_values) if counts_values else 0,
        "Interventions per MP (Max)": max(counts_values) if counts_values else 0,
        "Interventions per MP (Avg)": round(np.mean(counts_values), 2) if counts_values else 0,
        "Interventions per MP (Median)": round(np.median(counts_values), 2) if counts_values else 0,
        
        # Distribución por Intervención
        "Tokens per Intervention (Min)": min(stats_token_lengths) if stats_token_lengths else 0,
        "Tokens per Intervention (Max)": max(stats_token_lengths) if stats_token_lengths else 0,
        "Tokens per Intervention (Avg)": round(np.mean(stats_token_lengths), 2) if stats_token_lengths else 0,
    }

    print("\n[RESUMEN ESTADÍSTICO PRELIMINAR]")
    for k, v in statistics.items():
        print(f"  - {k}: {v}")

    # 4. Generar Universo de Oradores (Labels) y Vectorizar
    print(f"\nGenerando vectores para {total_mps} MPs...")
    
    sorted_mps = sorted(list(stats_mp_counts.keys()))
    mp_to_index = {mp: idx for idx, mp in enumerate(sorted_mps)}
    num_classes = len(sorted_mps)
    
    final_data = []
    
    for row in tqdm(all_processed_rows, desc="Vectorizing"):
        row_speakers = []
        row_interventions = []
        
        # Convertir mapa a listas paralelas
        for mp, texts in row['interventions_map'].items():
            row_speakers.append(mp)
            row_interventions.append(texts)
            
        # Crear One-Hot Vector
        label_vector = np.zeros(num_classes, dtype=int)
        for sp in row_speakers:
            if sp in mp_to_index:
                label_vector[mp_to_index[sp]] = 1
        
        final_data.append({
            'PK': row['PK'],
            'Text': row['Text'],
            'Speakers': row_speakers,
            'Interventions': row_interventions,
            'label': label_vector.tolist()
        })

    # 5. Guardado Final
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Guardando dataset HF en {OUTPUT_DIR}...")
    hf_dataset = Dataset.from_list(final_data)
    hf_dataset.save_to_disk(OUTPUT_DIR)
    
    # Guardar Mapeo
    with open(os.path.join(OUTPUT_DIR, "mp_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "id2label": {i: mp for i, mp in enumerate(sorted_mps)},
            "label2id": mp_to_index
        }, f, indent=4)
        
    # Guardar Estadísticas
    save_statistics(OUTPUT_DIR, statistics)

    print("\n[OK] Dataset y estadísticas generados correctamente.")

if __name__ == "__main__":
    main()