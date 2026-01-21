import re
import json
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict

FOLDER_DS = "/home/miguel/data/raw/CAN_dataset"
MAX_CHARACTERS = 40  # Limite de caracteres para procesar

def extract_interventions(texto) -> Dict[str, list]:
    # Diccionario para almacenar los resultados
    # Estructura: { "Nombre Diputado": ["Intervención 1", "Intervención 2"] }
    intervenciones = {}

    # 1. Definir el patrón de búsqueda (Regex)
    # Explicación:
    # (El señor|La señora) -> Grupo 1: Busca el prefijo de cortesía.
    # \s+ -> Espacios.
    # ([^:]+) -> Grupo 2: Captura todo hasta encontrar dos puntos (el nombre).
    # : -> El delimitador final del nombre.
    patron = r'(El señor|La señora)\s+([^:]+):'

    # Encontramos todas las coincidencias (iterador para obtener posiciones)
    matches = list(re.finditer(patron, texto))

    for i, match in enumerate(matches):
        # Nombre crudo (ej: "GALVÁN SASIA (desde su escaño)")
        nombre_crudo = match.group(2).strip()
        
        # 2. Limpieza del nombre para usarlo como clave consistente
        # Eliminamos "(desde su escaño)" pero dejamos nombres entre paréntesis como (Clavijo Batlle)
        nombre_limpio = nombre_crudo.replace("(desde su escaño)", "").strip()
        nombre_temp = nombre_limpio.replace("(Desde su escaño)", "").strip()

        match_nombre_real = re.search(r'\((.*?)\)', nombre_temp)
        
        if match_nombre_real:
            # Si encuentra paréntesis (ej: Clavijo Batlle), usa eso como clave
            nombre_limpio = match_nombre_real.group(1).strip()
        else:
            # Si no hay paréntesis (ej: GALVÁN SASIA), usa todo el texto
            nombre_limpio = nombre_temp

        nombre_limpio = nombre_limpio.title()  # Capitalizar cada palabra

        # Determinar dónde empieza y termina el discurso
        inicio_texto = match.end()
        
        # Si hay un siguiente orador, el texto termina donde empieza el siguiente match
        if i + 1 < len(matches):
            fin_texto = matches[i+1].start()
        else:
            # Si es el último, va hasta el final del string
            fin_texto = len(texto)

        # 3. Extraer y limpiar el contenido del discurso
        contenido = texto[inicio_texto:fin_texto].strip()

        # 4. Agrupar en el diccionario
        if nombre_limpio not in intervenciones:
            intervenciones[nombre_limpio] = []
        
        intervenciones[nombre_limpio].append(contenido)

    # 5. Convertir a JSON
    return intervenciones


def main():
    ds = load_from_disk(FOLDER_DS)['all']
    mp_interventions = {}


    for item in tqdm(ds, desc="Processing transcripts"):
        transcripts = item['text']    
        resultado_json = extract_interventions(transcripts)
        # borra las keys que sean "Presidenta" o "Presidente"
        for key in ["Presidenta", "Presidente"]:
            if key in resultado_json:
                del resultado_json[key]
        # borra aquellas keys que tenga mas de 40 caracteres
        keys_a_borrar = [key for key in resultado_json if len(key) > MAX_CHARACTERS]
        for key in keys_a_borrar:
            del resultado_json[key]
        for mp, interventions in resultado_json.items():
            if mp not in mp_interventions:
                mp_interventions[mp] = []
            mp_interventions[mp].extend(interventions)
    
    # borra aquellas keys que no tengan intervenciones
    keys_a_borrar = [key for key in mp_interventions if len(mp_interventions[key]) == 0]
    for key in keys_a_borrar:
        del mp_interventions[key]

    # borra aquellas intervenciones que sean vacías o solo espacios
    for mp in mp_interventions:
        mp_interventions[mp] = [interv for interv in mp_interventions[mp] if interv.strip() != ""]
    

    # Guardar el resultado en un archivo JSON
    with open('mp_interventions.json', 'w', encoding='utf-8') as f:
        json.dump(mp_interventions, f, ensure_ascii=False, indent=4)

    # sacar estadísticas básicas
    total_mps = len(mp_interventions)
    total_interventions = sum(len(interventions) for interventions in mp_interventions.values())
    print(f"Total MPs: {total_mps}")
    print(f"Total Interventions: {total_interventions}")
    # sacar estadisticas de la cantidad de intervenciones por MP
    intervenciones_por_mp = [len(interventions) for interventions in mp_interventions.values()]
    max_intervenciones = max(intervenciones_por_mp)
    min_intervenciones = min(intervenciones_por_mp)
    avg_intervenciones = total_interventions / total_mps
    print(f"Max Interventions by a MP: {max_intervenciones}")
    print(f"Min Interventions by a MP: {min_intervenciones}")
    print(f"Avg Interventions per MP: {avg_intervenciones:.2f}")
    # sacar estadísticas de la longitud de las intervenciones en tokens
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
    longitudes_tokens = []
    for interventions in tqdm(mp_interventions.values(), desc="Calculating token lengths"):
        for intervention in interventions:
            tokens = tokenizer.tokenize(intervention)
            longitudes_tokens.append(len(tokens))
    max_longitud = max(longitudes_tokens)
    min_longitud = min(longitudes_tokens)
    avg_longitud = sum(longitudes_tokens) / len(longitudes_tokens)
    print(f"Max Length of Interventions (tokens): {max_longitud}")
    print(f"Min Length of Interventions (tokens): {min_longitud}")
    print(f"Avg Length of Interventions (tokens): {avg_longitud:.2f}")

if __name__ == "__main__":
    main()

