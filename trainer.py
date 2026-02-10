import argparse
import json
import torch
import numpy as np
import random
import os
from datasets import load_from_disk
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType
)
from eval.Evaluator import Evaluator

from torch import nn

from trainers.StarSpaceTrainer import StarSpaceTrainer
from utils import SEED
from trainers.trainer_utils import initialize_determinism, compute_metrics, get_pos_weight_value, estimate_prior
from trainers.WeightedTrainer import WeightedTrainer
from trainers.nnPUTrainer import nnPUTrainer

# --- 1. CONFIGURACIÓN ---
parser = argparse.ArgumentParser(description="Baseline Dual (Train: Intervenciones / Test: Full Text)")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--mapping_path", type=str, required=True)
parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-small")
parser.add_argument("--mode", type=str, choices=["full", "lora", "qlora"], default="full")
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--intervention_strategy", type=str, choices=["concat", "split"], default="split")
parser.add_argument("--label_strategy", type=str, choices=["author", "all_participants"], default="all_participants")
parser.add_argument("--tiny", action="store_true", help="Usar un subset tiny para pruebas rápidas")
parser.add_argument("--dapt_adapter_path", type=str, default=None, help="Ruta a un adaptador LoRA entrenado en DAPT (opcional)")

# type of loss
parser.add_argument("--loss_type", type=str, choices=["bce", "nnpuloss", "starspace"], default="bce")
parser.add_argument("--num_negatives", type=int, default=5, help="Number of negative samples for StarSpace loss")

# general hyperparameters
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--pos_weight_type", type=str, choices=["linear", "sqrt", "log"], default="sqrt", help="Type of pos_weight to use")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate to use (overrides defaults)")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")

# lora/qlora specific
parser.add_argument("--lora_r", type=int, default=16, help="Lora rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="Lora dropout")

args = parser.parse_args()
initialize_determinism(args.seed)

model_name_base = os.path.basename(args.model_name)
dataset_base = os.path.basename(args.data_path)

if args.loss_type == "bce":
    trainer_name = f"BCETrainer_{args.pos_weight_type}"
elif args.loss_type == "starspace":
    trainer_name = f"StarSpaceTrainer_k{args.num_negatives}"
else:
    trainer_name = f"nnPUTrainer"

full_experiment_name = (
    f"{model_name_base}_{args.mode}_"
    f"{trainer_name}_"
    f"epochs{args.epochs}_"
    f"bs{args.batch_size}_"
    f"lr{args.learning_rate}_"
    f"maxlen{args.max_length}"
)
folder_to_save_results = os.path.join(args.output_dir, dataset_base, full_experiment_name)

# --- 2. CARGA DE DATOS ---
print(f"[-] Cargando mapeo...")
with open(args.mapping_path, 'r') as f:
    mapping_data = json.load(f)

id2label = {int(k): v for k, v in mapping_data["id2label"].items()}
label2id = mapping_data["label2id"]
num_labels = len(id2label)

print(f"[-] Cargando dataset...")
raw_dataset = load_from_disk(args.data_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# --- 3. PREPROCESAMIENTO DIFERENCIADO ---

# A) FUNCIÓN PARA TRAIN (Explosión: 1 Intervención -> 1 MP)

def process_train_function(batch):
    new_texts = []
    new_labels = []
    
    for i in range(len(batch['PK'])):
        speakers = batch['Speakers'][i] # Lista de todos los participantes del doc
        interventions = batch['Interventions'][i] # Lista de listas de textos
        
        if len(speakers) != len(interventions): continue

        # --- OPTIMIZACIÓN: Pre-calcular el vector del documento completo ---
        # Si la estrategia es 'all_participants', el vector Y es idéntico para todas 
        # las frases de este documento. Lo calculamos una vez aquí.
        doc_wide_label_vec = None
        if args.label_strategy == "all_participants":
            doc_wide_label_vec = [0.0] * num_labels
            for sp in speakers:
                if sp in label2id:
                    doc_wide_label_vec[label2id[sp]] = 1.0

        # Iteramos sobre cada orador y sus textos
        for speaker_name, intervention_parts in zip(speakers, interventions):
            if speaker_name not in label2id: continue
            
            # 1. Preparar TEXTO (Split vs Concat)
            texts_to_process = []
            if args.intervention_strategy == "concat":
                full_text = " ".join(intervention_parts)
                if len(full_text.strip()) > 5: 
                    texts_to_process.append(full_text)
            elif args.intervention_strategy == "split":
                texts_to_process = [t for t in intervention_parts if len(t.strip()) > 5]

            # 2. Preparar ETIQUETA (Author vs All Participants)
            if args.label_strategy == "all_participants":
                # Nueva estrategia: Usamos el vector común del documento
                label_vec = doc_wide_label_vec
            else:
                # Estrategia clásica ("author"): Solo el orador actual es 1
                label_vec = [0.0] * num_labels
                label_vec[label2id[speaker_name]] = 1.0
            
            # 3. Generar muestras
            for text in texts_to_process:
                new_texts.append(text)
                new_labels.append(label_vec)
    
    tokenized = tokenizer(new_texts, padding="max_length", truncation=True, max_length=args.max_length)
    tokenized["labels"] = new_labels
    return tokenized
# B) FUNCIÓN PARA DEV/TEST (Full Text: 1 Doc -> N MPs)

def process_eval_function(batch):
    # Aquí usamos directamente 'Text' y 'Speakers' del batch original
    texts = batch['Text']
    batch_speakers = batch['Speakers']
    
    final_labels = []
    
    for speakers_list in batch_speakers:
        # MULTI-LABEL (Varios 1s correspondientes a todos los speakers)
        label_vec = [0.0] * num_labels
        for sp in speakers_list:
            if sp in label2id:
                label_vec[label2id[sp]] = 1.0
        final_labels.append(label_vec)
    
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length)
    tokenized["labels"] = final_labels
    return tokenized

print("[-] Procesando TRAIN (Intervenciones individuales)...")
split_to_use = 'train' if not args.tiny else 'tiny'
train_dataset = raw_dataset[split_to_use].map(
    process_train_function, 
    batched=True, 
    remove_columns=raw_dataset[split_to_use].column_names
)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("[-] Procesando DEV y TEST (Documentos completos)...")
# Nota: Aquí NO hacemos 'explode', usamos el dataset original
dev_dataset = raw_dataset['dev'].map(
    process_eval_function, 
    batched=True, 
    remove_columns=raw_dataset['dev'].column_names
)
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset = raw_dataset['test'].map(
    process_eval_function, 
    batched=True, 
    remove_columns=raw_dataset['test'].column_names
)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"    Train size: {len(train_dataset)} (Intervenciones)")
print(f"    Dev size:   {len(dev_dataset)} (Documentos)")
print(f"    Test size:  {len(test_dataset)} (Documentos)")

# --- 4. MODELO ---
print(f"[-] Inicializando modelo ({args.mode})...")
model_config = {
    "num_labels": num_labels,
    "id2label": id2label,
    "label2id": label2id,
    "problem_type": "multi_label_classification" # Vital para que funcione en ambos modos
}

if args.mode == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto", **model_config)
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, device_map="auto", **model_config)

if args.dapt_adapter_path:
    from peft import PeftModel
    print(f"[-] Cargando adaptador DAPT desde {args.dapt_adapter_path}...")
    model = PeftModel.from_pretrained(model, args.dapt_adapter_path)
    print("[-] Fusionando adaptador DAPT en el modelo base...")
    model = model.merge_and_unload()
    folder_to_save_results += folder_to_save_results + "_dapt"


if args.mode in ["lora", "qlora"]:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_config)
    # imprime la cantidad de parametros entrenables
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[-] Parámetros totales: {total_params}, Entrenables: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
# --- 5. ENTRENAMIENTO ---

training_args = TrainingArguments(
    output_dir=folder_to_save_results,
    learning_rate=args.learning_rate if args.learning_rate else (2e-4 if args.mode != "full" else 2e-5),
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size, # Ahora podemos usar batch mayor en eval si queremos
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,  # Al final, cargar el mejor modelo según F1
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    save_total_limit=2,           # Guardar solo los 2 mejores checkpoints para ahorrar espacio
    fp16=True,
    seed=args.seed,
    data_seed=args.seed,
    logging_steps=100,
    report_to="none"
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=args.early_stopping_patience,
)

if args.loss_type == "bce":
    pos_weight_value = get_pos_weight_value(args.pos_weight_type, num_labels)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        pos_weight_value=pos_weight_value,
        callbacks=[early_stopping_callback]
    )
elif args.loss_type == "nnpuloss":
    estimated_pi = estimate_prior(raw_dataset[split_to_use])
    print(f"    Usando prior estimado pi = {estimated_pi:.5f} para nnPU Loss.")
    trainer = nnPUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        prior=estimated_pi,
        callbacks=[early_stopping_callback],
        gamma=1.0,          # Penalización estándar
        beta=0.0            # Umbral estándar
    )
elif args.loss_type == "starspace":
    def compute_metrics_scaled(p):
        predictions, labels = p
        
        # --- FIX CRÍTICO: ESCALADO ---
        # Recuperamos la dimensión del modelo desde el scope global o hardcodeado
        # (Qwen suele ser 1024 o 896 dependiendo de la versión, mejor leerlo de config)
        hidden_dim = model.config.hidden_size 
        scale_factor = np.sqrt(hidden_dim)
        
        # Dividimos los logits gigantes antes de la sigmoide
        predictions = predictions / scale_factor
        # -----------------------------

        # Convertimos logits a probabilidades
        probs = 1 / (1 + np.exp(-predictions))
        
        # Usamos un umbral estándar de 0.5 para monitorear
        y_pred = (probs > 0.5).astype(int)
        
        # Calculamos métricas
        f1 = f1_score(labels, y_pred, average='micro', zero_division=0)
        precision = precision_score(labels, y_pred, average='micro', zero_division=0)
        recall = recall_score(labels, y_pred, average='micro', zero_division=0)
        
        return {
            'f1_micro': f1,
            'precision': precision, 
            'recall': recall
        }
    trainer = StarSpaceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_scaled,
        num_negatives=args.num_negatives,
        callbacks=[early_stopping_callback]
    )
else:
    raise ValueError(f"Tipo de loss desconocido: {args.loss_type}")

print("[-] Entrenando...")
trainer.train()
trainer.save_model(f"{folder_to_save_results}/final_model")

# --- 6. OPTIMIZACIÓN DE UMBRAL (DEV) ---
print("\n[-] Prediciendo en DEV (Full Text)...")
# Al ser clasificación estándar (1 fila = 1 doc), trainer.predict funciona directo
dev_predictions = trainer.predict(dev_dataset)
if args.loss_type == "starspace":
    hidden_size = model.config.hidden_size
    dev_logits = dev_predictions.predictions / (hidden_size ** 0.5)
else:
    dev_logits = dev_predictions.predictions
dev_labels = dev_predictions.label_ids

# Sigmoide
y_score_dev = 1 / (1 + np.exp(-dev_logits))

print("[-] Optimizando umbral...")
best_f1 = 0.0
best_threshold = 0.5
thresholds = np.arange(0.1, 0.95, 0.05)

for th in thresholds:
    y_pred = (y_score_dev >= th).astype(int)
    # Calculamos F1 sobre la matriz completa directamente
    from sklearn.metrics import f1_score
    f1 = f1_score(dev_labels, y_pred, average='micro', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print(f"    Mejor umbral: {best_threshold:.2f} (F1 Dev: {best_f1:.4f})")

# --- 7. EVALUACIÓN FINAL (TEST) ---
print(f"\n[-] Evaluando en TEST (Full Text) con umbral {best_threshold:.2f}...")
test_predictions = trainer.predict(test_dataset)
if args.loss_type == "starspace":
    hidden_size = model.config.hidden_size
    y_score_test = test_predictions.predictions / (hidden_size ** 0.5)
else:
    y_score_test = test_predictions.predictions
y_score_test = 1 / (1 + np.exp(-y_score_test))  # Sigmoide
y_true_test = test_predictions.label_ids

# Llamada a tu Evaluator
evaluator = Evaluator()
metrics = evaluator.compute_all_metrics(y_true_test, y_score_test, threshold=best_threshold)
evaluator.print_report(metrics)

# Guardar resultados
json_path = os.path.join(folder_to_save_results, "test_metrics.json")
with open(json_path, "w") as f:
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    json.dump(serializable_metrics, f, indent=4)

sorted_ids = sorted([int(k) for k in id2label.keys()])
mp_names_ordered = [id2label[k] for k in sorted_ids]

results_data = {
    "model_name": full_experiment_name,  # Nombre descriptivo
    "args": vars(args),                  # Guardamos TODOS los argumentos por si acaso
    "threshold": best_threshold,         # El umbral óptimo calculado en Dev
    "metrics": metrics,                  # Diccionario de métricas (F1, MAP, nDCG...)
    "mp_names": mp_names_ordered,        # Lista de nombres [MP_0, MP_1, ...]
    "y_true": y_true_test,               # Matriz numpy (N_docs x N_mps)
    "scores": y_score_test               # Matriz numpy (N_docs x N_mps)
}

import pickle
pickle_path = os.path.join(folder_to_save_results, "full_results.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(results_data, f)

print(f"\n[-] Métricas guardadas en: {json_path}")
print(f"[-] Objeto completo (Pickle) guardado en: {pickle_path}")
print(f"    (Nombre del modelo registrado: {full_experiment_name})")