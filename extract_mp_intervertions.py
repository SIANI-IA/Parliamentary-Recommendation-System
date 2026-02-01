import re
import json
import os
import numpy as np
import csv
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Any
import unicodedata

# --- CONFIGURACIÓN ---
FOLDER_DS = "/home/miguel/data/raw/CAN_dataset"
OUTPUT_DIR = "dataset/parcanDeb-rec"
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
MIN_CHAR_CONTENT = 300

# PALABRAS PROHIBIDAS (BLACKLIST)
# Si una captura contiene CUALQUIERA de estas palabras, se descarta.
FORBIDDEN_WORDS = {
    # Verbos y conectores de narración
    "dijo", "dice", "decía", "decia", "manifestó", "manifesto", 
    "hacía", "hacia", "referencia", "alerta", "principio", 
    "porque", "pero", "cuando", "donde", "como", "que", 
    "permítame", "permitame", "diga", "digo", "mira", "mire", 
    "favor", "gracias", "palabra", "turno", "intervención",
    "puso", "cosa", "cosas", "estado", "competencialmente",
    "empiezo", "inicio", "hablaba", "planteaba", "antes",
    "quedan", "quedaba", "segundos", "minutos", "tiempo",
    
    # Palabras sueltas detectadas en tu lista
    "cierto", "lapsus", "victoria", "tema", "fundamental",
    "siguiente", "bien", "evidencia", "titulares", 
    
    # Ruido técnico o de ubicación
    "telemáticamente", "telematicamente", "desde", "escaño", 
    "sede", "fuera", "subinspector", "zacarías" # (Zacarías parece ruido suelto si no tiene apellido)
}

def clean_speaker_name_universal(raw_segment):
    """
    Limpieza, validación y filtrado estricto.
    """
    # 1. ELIMINACIÓN DE RUIDO TÉCNICO INICIAL
    # Quitamos paréntesis de escaño
    match_escaño = re.search(r'\s*\(?\s*desde su escaño.*', raw_segment, flags=re.IGNORECASE)
    if match_escaño:
        raw_segment = raw_segment[:match_escaño.start()]
    
    # Quitamos guiones de metadatos (Ej: "Paulino Rivero -Titulares-")
    if "-" in raw_segment:
        raw_segment = raw_segment.split("-")[0]

    segment = raw_segment.strip()

    # 2. CORRECCIÓN DE REPETICIONES ("El señor X. El señor Y")
    split_pattern = r'(?:El señor|La señora|Don|Doña)\s+'
    matches = list(re.finditer(split_pattern, segment, flags=re.IGNORECASE))
    
    if matches:
        last_match = matches[-1]
        candidate = segment[last_match.end():].strip()
        if len(candidate) > 2:
            segment = candidate

    # 3. ESTRATEGIA DE PARÉNTESIS
    match_parens = re.search(r'\((.*?)\)', segment)
    if match_parens:
        candidate = match_parens.group(1).strip()
        if len(candidate) > 2:
            segment = candidate
        else:
            segment = re.sub(r'\(.*?\)', '', segment)

    # 4. LIMPIEZA DE PUNTUACIÓN Y NORMALIZACIÓN
    clean_text = segment.replace(":", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").strip()
    clean_text = " ".join(clean_text.split()).title()
    
    # 5. CORRECCIONES MANUALES DE OCR Y CARGOS
    # Arregla "Preside Nte" -> "Presidente"
    if "Preside Nte" in clean_text or "Preside" == clean_text:
        return None
    
    # Unificamos Vicepresidentas con Presidente/a (actúan como moderadores)
    if "Vicepresidenta" in clean_text or "President" in clean_text:
        return None
    
    # --- ZONA DE VALIDACIÓN (FILTROS) ---
    
    words = clean_text.split()
    
    # A. Filtro de Longitud de Palabras
    # "Carmen Rosa Hablaba De La Victoria" -> 6 palabras. Límite estricto a 5.
    if len(words) > 5:
        return None
        
    # B. Filtro de Palabras Prohibidas (La parte más importante)
    for w in words:
        if w.lower() in FORBIDDEN_WORDS:
            return None

    # C. Filtros de seguridad básicos
    # Menos de 3 letras o nombres de pila sueltos sospechosos (Ej: "Ana", "Mario")
    # Si solo tiene 1 palabra y es un nombre común sin apellido, riesgo alto de ser mención.
    if len(clean_text) < 4: 
        return None
        
    return clean_text

def extract_interventions(texto: str) -> Dict[str, list]:
    intervenciones = {}
    
    # Regex: Busca "El señor/La señora" tras punto o inicio.
    patron = r'(?:(?<=\.)|^)\s*(?:El señor|La señora|Don|Doña)\s+([^:]{2,100}):'
    
    matches = list(re.finditer(patron, texto, flags=re.MULTILINE))

    for i, match in enumerate(matches):
        raw_captured = match.group(1)
        
        speaker_clean = clean_speaker_name_universal(raw_captured)
        
        if not speaker_clean:
            continue
            
        start_idx = match.end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(texto)
        
        contenido = texto[start_idx:end_idx].strip()
        
        if len(contenido) >= MIN_CHAR_CONTENT:
            if speaker_clean not in intervenciones:
                intervenciones[speaker_clean] = []
            intervenciones[speaker_clean].append(contenido)

    return intervenciones

def save_statistics(output_path: str, stats: Dict[str, Any]):
    with open(os.path.join(output_path, "stats_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "stats_summary.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            writer.writerow([k, v])

def main():
    print(f"Cargando dataset desde {FOLDER_DS}...")
    try:
        ds = load_from_disk(FOLDER_DS)['all']
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    print(f"Cargando tokenizador ({MODEL_ID})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except:
        print(" [!] Tokenizador no disponible. Usando conteo simple.")
        tokenizer = None

    processed_rows = []
    stats_mp_counts = {}
    stats_token_lengths = [] 
    
    print("Procesando documentos...")
    
    for item in tqdm(ds, desc="Extraction"):
        original_text = item.get('text', '')
        original_pk = item.get('pk', item.get('PK', str(hash(original_text[:50]))))
        
        interventions_dict = extract_interventions(original_text)
        
        row_speakers = []
        row_interventions = []

        for mp, texts in interventions_dict.items():
            stats_mp_counts[mp] = stats_mp_counts.get(mp, 0) + len(texts)
            for text in texts:
                length = len(tokenizer.encode(text)) if tokenizer else len(text.split())
                stats_token_lengths.append(length)

            row_speakers.append(mp)
            row_interventions.append(texts)
        
        if len(row_speakers) > 0:
            processed_rows.append({
                'PK': original_pk,
                'Text': original_text,
                'Speakers': row_speakers,
                'Interventions': row_interventions
            })

    counts_values = list(stats_mp_counts.values())
    total_mps_unique = len(stats_mp_counts)
    
    statistics = {
        "Total Documents": len(processed_rows),
        "Total Unique MPs": total_mps_unique,
        "Total Interventions": sum(counts_values) if counts_values else 0,
        "Total Tokens": sum(stats_token_lengths),
        "Avg Tokens per Intervention": round(np.mean(stats_token_lengths), 2) if stats_token_lengths else 0,
        "Min Tokens per Intervention": min(stats_token_lengths) if stats_token_lengths else 0,
        "Max Tokens per Intervention": max(stats_token_lengths) if stats_token_lengths else 0,
        "Avg Interventions per MP": round(np.mean(counts_values), 2) if counts_values else 0
    }

    print("\n[RESUMEN ESTADÍSTICO]")
    for k, v in statistics.items():
        print(f"  - {k}: {v}")
        
    sorted_mps = sorted(list(stats_mp_counts.keys()))
    mp_to_index = {mp: idx for idx, mp in enumerate(sorted_mps)}
    num_classes = len(sorted_mps)
    
    print(f"\nGenerando vectores para {num_classes} oradores únicos...")

    final_data = []
    for row in tqdm(processed_rows, desc="Vectorizing"):
        label_vector = np.zeros(num_classes, dtype=int)
        for speaker in row['Speakers']:
            if speaker in mp_to_index:
                label_vector[mp_to_index[speaker]] = 1
        
        row['label'] = label_vector.tolist()
        final_data.append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_dataset = Dataset.from_list(final_data)
    hf_dataset.save_to_disk(OUTPUT_DIR)
    
    with open(os.path.join(OUTPUT_DIR, "mp_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "id2label": {i: mp for i, mp in enumerate(sorted_mps)},
            "label2id": mp_to_index
        }, f, ensure_ascii=False, indent=4)

    save_statistics(OUTPUT_DIR, statistics)
    print("\n[OK] Completado.")

if __name__ == "__main__":
    main()