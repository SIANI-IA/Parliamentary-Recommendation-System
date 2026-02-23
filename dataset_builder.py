import json
from datasets import Dataset, DatasetDict
import pandas as pd

class ParliamentaryDataBuilder:
    def __init__(self, original_dataset_dict: DatasetDict, mapping_json_path: str, lang: str = 'spanish'):
        """
        Inicializa el constructor con tu DatasetDict original.
        """
        assert lang in ['spanish', 'english'], "Currently only 'spanish' and 'english' are supported."

        self.dataset = original_dataset_dict
        self.lang = lang

        # 1. Cargar el mapeo oficial desde tu archivo JSON
        print(f"Cargando mapeo de MPs desde: {mapping_json_path}")
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            
        # Nos quedamos con label2id (Nombre -> ID)
        self.label2id = mapping_data['label2id']
        print(f"Total de MPs cargados desde JSON: {len(self.label2id)}")
        
        # Configuración de Prompts (Fácil de modificar para tus experimentos)
        self.system_prompt = (
            "Eres un asistente parlamentario experto. Analiza el siguiente documento "
            "legislativo y genera una lista ordenada con los identificadores de los "
            "parlamentarios (MPs) más relevantes, separados por comas."
        )
        
    def _format_mp_token(self, speaker_name):
        """
        Convierte el nombre real en su token especial usando el diccionario oficial.
        Ejemplo: "Abrante Brito" -> "[MP_0]"
        """        
        # Buscamos su ID numérico
        if speaker_name in self.label2id:
            mp_index = self.label2id[speaker_name]
            return f"[MP_{mp_index}]"
        else:
            # Fallback de seguridad (no debería ocurrir si el JSON está completo)
            print(f"Advertencia: '{speaker_name}' no encontrado en el JSON.")
            return "[MP_UNK]"

    # ==========================================
    # FASE 1: ENTRENAMIENTO AUTOREGRESIVO
    # ==========================================
    def _flatten_autoregressive_batch(self, batch):
        """
        Transforma 1 documento en N ejemplos (uno por cada intervención).
        """
        new_texts = []
        
        # Iteramos sobre cada documento del batch
        for text, speakers, interventions in zip(batch['Text'], batch['Speakers'], batch['Interventions']):
            # Asumimos que speakers e interventions tienen la misma longitud
            for speaker, speaker_interventions in zip(speakers, interventions):
                speaker_token = self._format_mp_token(speaker)
                title = text.split(".")[1] if self.lang == 'spanish' else text.split("\n")[0]
                
                # Un speaker puede tener varias intervenciones en un mismo documento
                for intervention in speaker_interventions:
                    # Formato estricto para que el modelo asocie al MP con su texto
                    text_block = (
                        f"[Iniciativa]: {title.strip()}\n"
                        f"[Orador]: {speaker_token}\n"
                        f"[Intervención]: {intervention.strip()}"
                    )
                    new_texts.append(text_block)
                    
        return {"text": new_texts}

    def build_autoregressive_dataset(self) -> DatasetDict:
        """
        Construye el dataset para el Continuous Pre-training.
        Aplica la transformación a train, dev y test.
        """
        print("Construyendo dataset Autoregresivo...")
        # Usamos batched=True y eliminamos las columnas viejas porque 
        # pasamos de N filas a N*X filas.
        ar_dataset = self.dataset.map(
            self._flatten_autoregressive_batch,
            batched=True,
            remove_columns=self.dataset['train'].column_names,
            desc="Aplanando intervenciones"
        )
        return ar_dataset

    # ==========================================
    # FASE 2: INSTRUCTION TUNING (RANKING)
    # ==========================================
    def _format_instruction_batch(self, batch):
        """
        Transforma 1 documento en 1 prompt de instrucción (Listwise Ranking).
        """
        system_prompt = []
        document = []
        answer = []
        
        for text, speakers in zip(batch['Text'], batch['Speakers']):
            # 1. Preparar la lista de MPs objetivo (Ground Truth)
            # NOTA: Aquí es donde en el futuro puedes inyectar tus Hard Negatives (PUL)
            target_mps = [self._format_mp_token(spk) for spk in speakers]
            target_string = ", ".join(target_mps)
            
            # 2. Truncamiento del texto (Opcional, pero recomendado por memoria)
            # Si quieres hacerlo por palabras/caracteres antes del tokenizador
            # text_truncated = " ".join(text.split()[:800]) 
            
            # 3. Construir el prompt completo estilo Chat/Instruct
            # (Puedes adaptar esto si usas el formato ChatML de Llama-3 o Qwen3)
            system_prompt.append(self.system_prompt)
            document.append(text)
            answer.append(target_string)
            
        return {
            "system_prompt": system_prompt,
            "document": document,
            "answer": answer
        }

    def build_instruction_dataset(self) -> DatasetDict:
        """
        Construye el dataset para el fine-tuning supervisado (Generar la lista).
        """
        print("Construyendo dataset de Instrucción...")
        # Mantenemos la estructura 1 a 1, solo añadimos la columna con el prompt
        instruct_dataset = self.dataset.map(
            self._format_instruction_batch,
            batched=True,
            remove_columns=self.dataset['train'].column_names,
            desc="Generando prompts de instrucción"
        )
        return instruct_dataset

# ==========================================
# CÓMO USARLO
# ==========================================
if __name__ == "__main__":
    from datasets import load_from_disk
    # Supongamos que tu dataset se llama 'dataset_dict_original'
    dataset_name = "parcanDeb-rec-split" # Ajusta esto a tu dataset
    dataset_path = f"./dataset/{dataset_name}"
    dataset_dict_original = load_from_disk(dataset_path) # Ajusta el path a tu dataset
    mapping_json_path = f"{dataset_path}/mp_mapping.json" # Ajusta el path a tu JSON
    
    # 1. Instanciar el builder
    builder = ParliamentaryDataBuilder(dataset_dict_original, mapping_json_path, lang='spanish') # Cambia a 'spanish' si tu dataset está en español
    
    # 2. Generar y guardar el dataset Autoregresivo (Fase 1)
    ar_dataset = builder.build_autoregressive_dataset()
    #ar_dataset.save_to_disk("./data/autoregressive_split")
    print(f"Autoregresivo Train filas: {len(ar_dataset['train'])}") # Ej: ~54,000 intervenciones
    # ver ejemplo de una fila
    print(ar_dataset['train'][1]['text'])
    
    # 3. Generar y guardar el dataset de Instrucción (Fase 2)
    instruct_dataset = builder.build_instruction_dataset()
    print(f"Instructivo Train filas: {len(instruct_dataset['train'])}") # Ej: 7,640 documentos
    print(instruct_dataset['train']) # Ver ejemplo de prompt


    #instruct_dataset.save_to_disk("./data/instruction_split")