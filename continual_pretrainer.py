import os
import argparse
import torch
from itertools import chain
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from utils import SEED
from trainers.trainer_utils import initialize_determinism

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data_path", type=str, required=True, help="Ruta a tu dataset HF (carpeta)")
    parser.add_argument("--output_dir", type=str, default="./results/cp/dapt")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--tiny", action="store_true", help="Usar un modelo pequeño para pruebas rápidas")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    model_base_name = args.model_name.split("/")[-1]
    print(f"[-] Modelo base: {model_base_name}")

    initialize_determinism(args.seed)

    dataset_name = os.path.basename(args.data_path)
    output_dir = args.output_dir + "_" + model_base_name + "_maxlen" + str(args.max_length) + "_" + dataset_name
    if args.tiny:
        output_dir += "_tiny"
    os.makedirs(output_dir, exist_ok=True)

    # Configuración para torchrun/DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    print(f"[-] Cargando modelo base: {args.model_name}")
    
    # 1. Modelo y Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        dtype=torch.bfloat16, # H100 nativo
        #attn_implementation="flash_attention_2"
    )

    # 2. Configurar LoRA para Causal LM
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", # Importante: no es SEQ_CLS
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()

    # 3. Carga y Procesamiento del Dataset (Directo, sin TXT)
    print(f"[-] Cargando dataset desde disco: {args.data_path}")
    dataset = load_from_disk(args.data_path)
    
    # Asumimos que usas el split 'train' para aprender
    if args.tiny:
        print("[-] Usando subset tiny para pruebas rápidas")
        train_dataset = dataset["train"].shuffle(seed=SEED).select(range(100)) # Solo 100 ejemplos
    else:
        train_dataset = dataset["train"]
    
    # Paso A: Tokenizar los textos crudos
    column_names = train_dataset.column_names
    text_column = "Text" if "Text" in column_names else "text" # Ajusta si tu columna se llama diferente

    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    print("[-] Tokenizando...")
    tokenized_datasets = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(), # Usa todas las CPUs disponibles
        remove_columns=column_names # Borramos texto crudo para ahorrar RAM
    )

    # Paso B: Agrupar en bloques (Main chunking logic)
    # Esto convierte 1000 documentos de longitudes variadas en X bloques de exactamente 'block_size'
    block_size = args.max_length

    def group_texts(examples):
        # Concatenar todos los textos del batch
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Descartamos el resto pequeño que no llega al block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Partir en trozos de block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Etiqueta = Input desplazado (clásico Next Token Prediction)
        # El DataCollatorForLanguageModeling lo hace automático, pero aquí aseguramos estructura
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"[-] Agrupando textos en bloques de {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )
    
    if local_rank == 0:
        print(f"[-] Total de bloques para entrenar: {len(lm_datasets)}")

    # 4. Entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=1, # 8*4 = 32 batch size efectivo
        learning_rate=5e-5,            # LR muy conservador
        num_train_epochs=3,            # Suficiente para adaptar, no memorizar
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        #run_name="dapt_llama_canada_direct"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[-] Iniciando entrenamiento DAPT...")
    trainer.train()

    print("[-] Guardando adaptador...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()