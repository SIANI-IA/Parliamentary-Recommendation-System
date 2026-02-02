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
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    set_seed
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType
)
from eval.Evaluator import Evaluator
from utils import SEED

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 0. REPRODUCIBILIDAD ---
def initialize_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #torch.use_deterministic_algorithms(True, warn_only=True)

# --- 1. CONFIGURACIÓN ---
parser = argparse.ArgumentParser(description="Baseline Dual (Train: Intervenciones / Test: Full Text)")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--mapping_path", type=str, required=True)
parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-small")
parser.add_argument("--mode", type=str, choices=["full", "lora", "qlora"], default="full")
parser.add_argument("--output_dir", type=str, default="./results_dual")
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)

args = parser.parse_args()
initialize_determinism(args.seed)

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
        speakers = batch['Speakers'][i]
        interventions = batch['Interventions'][i]
        
        if len(speakers) != len(interventions): continue

        for speaker_name, intervention_parts in zip(speakers, interventions):
            if speaker_name not in label2id: continue
            
            full_text = " ".join(intervention_parts)
            if len(full_text.strip()) < 10: continue

            # SINGLE-LABEL (Solo un 1)
            label_vec = [0.0] * num_labels
            speaker_idx = label2id[speaker_name]
            label_vec[speaker_idx] = 1.0
            
            new_texts.append(full_text)
            new_labels.append(label_vec)
            
    # Tokenización "al vuelo" para ahorrar memoria
    tokenized = tokenizer(new_texts, padding="max_length", truncation=True, max_length=args.max_length)
    tokenized["labels"] = new_labels
    return tokenized

# B) FUNCIÓN PARA DEV/TEST (Full Text: 1 Doc -> N MPs)
# --- BLOQUE NUEVO: MÉTRICAS PARA EL TRAINING LOOP ---

def compute_metrics(p):
    predictions, labels = p
    # Convertimos logits a probabilidades
    probs = 1 / (1 + np.exp(-predictions))
    
    # Usamos un umbral estándar de 0.5 para monitorear durante el entrenamiento
    # (El umbral óptimo se calculará al final, esto es solo para ver progreso)
    y_pred = (probs > 0.5).astype(int)
    
    # Calculamos F1 Micro (la métrica más importante para ver convergencia global)
    f1 = f1_score(labels, y_pred, average='micro', zero_division=0)
    precision = precision_score(labels, y_pred, average='micro', zero_division=0)
    recall = recall_score(labels, y_pred, average='micro', zero_division=0)
    
    return {
        'f1_micro': f1,
        'precision': precision, 
        'recall': recall
    }

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
train_dataset = raw_dataset['train'].map(
    process_train_function, 
    batched=True, 
    remove_columns=raw_dataset['train'].column_names
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

if args.mode in ["lora", "qlora"]:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, peft_config)

# --- 5. ENTRENAMIENTO ---

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # FOCAL LOSS IMPLEMENTATION
        # 1. Sigmoide para obtener probabilidades
        probs = torch.sigmoid(logits)
        
        # 2. Definir target (labels)
        targets = labels.float()
        
        # 3. Calcular la Loss por componente
        # p_t es la probabilidad asociada a la clase correcta
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Factor de ponderación (alpha para balancear clases, gamma para hard examples)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # BCE standard (sin reducción para poder multiplicar)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal Loss final
        loss = (alpha_factor * modulating_factor * bce_loss).mean()
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=2e-4 if args.mode != "full" else 2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size, # Ahora podemos usar batch mayor en eval si queremos
    num_train_epochs=args.epochs,
    weight_decay=0.01,
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

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    eval_dataset=dev_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print("[-] Entrenando...")
trainer.train()
trainer.save_model(f"{args.output_dir}/final_model")

# --- 6. OPTIMIZACIÓN DE UMBRAL (DEV) ---
print("\n[-] Prediciendo en DEV (Full Text)...")
# Al ser clasificación estándar (1 fila = 1 doc), trainer.predict funciona directo
dev_predictions = trainer.predict(dev_dataset)
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
y_score_test = 1 / (1 + np.exp(-test_predictions.predictions))
y_true_test = test_predictions.label_ids

# Llamada a tu Evaluator
evaluator = Evaluator()
metrics = evaluator.compute_all_metrics(y_true_test, y_score_test, threshold=best_threshold)
evaluator.print_report(metrics)

# Guardar resultados
with open(f"{args.output_dir}/test_metrics.json", "w") as f:
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    json.dump(serializable_metrics, f, indent=4)