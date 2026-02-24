from datasets import load_from_disk, Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from unsloth import UnslothTrainer, UnslothTrainingArguments
from trl import SFTTrainer, SFTConfig
import wandb
import os
import torch

from utils import SEED

dataset_name = "parcanDeb-rec-split" # Ajusta esto a tu dataset
FOLDER_KNOWLEDGE = f"dataset/autoregressive_split/{dataset_name}"
FOLDER_QA = f"dataset/instruction_split/{dataset_name}"
MODEL_NAME = "Qwen/Qwen3-0.6B"
is_sync = True
is_adapter = False # TODO: solucionar esto no funciona
TINY = False # Para pruebas rápidas, setea a False para entrenar con todo el dataset.
BATCH_SIZE = 2
MAX_LENGTH = 1024
LOAD_IN_4BIT = False
SUPER_EPOCHS = 5
RANK_LORA = 16
model_basename = MODEL_NAME.split("/")[-1]
USE_WANDB = True
FOLDER_TO_SAVE_MODELS = "./results/testing"
name_dataset = FOLDER_QA.split("/")[-1]
name_of_folder_model = os.path.join(FOLDER_TO_SAVE_MODELS, f"{model_basename}-r{RANK_LORA}-ccf-{name_dataset}")

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_LENGTH,
        full_finetuning = False,
        load_in_4bit = LOAD_IN_4BIT,
        load_in_8bit = False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    if not is_adapter:
        model = FastLanguageModel.get_peft_model(
            model,
            r = RANK_LORA,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj", 
                #"lm_head", "embed_tokens"
            ],
            lora_alpha = RANK_LORA*2,
            lora_dropout = 0, 
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = SEED,
            use_rslora = False,
            loftq_config = None,
        )
    else:
        print("The model is loaded with adapters.")
    model.print_trainable_parameters()
    return model, tokenizer



def build_prompt_it(tokenizer, system_prompt: str, prompt: str, response: str) -> str:
    """Builds the chat prompt for a single example using the tokenizer chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )

def generate_qa_prompts(dataset, tokenizer):
    prompts = []
    for item in dataset:
        system_prompt = item["system_prompt"]
        prompt = item["document"][0:2000] # TODO: ajustar esto para que no corte información relevante, quizás usando el tokenizador para truncar por tokens en lugar de por caracteres.
        response = item["answer"]
        prompts.append(build_prompt_it(tokenizer, system_prompt, prompt, response))
    return prompts

def generate_qa_prompts_for_testing(dataset, tokenizer):

    def build_prompt_testing(tokenizer, system_prompt: str, prompt: str) -> str:
        """Builds the chat prompt for a single example using the tokenizer chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    
    prompts = []
    for item in dataset:
        system_prompt = item["system_prompt"]
        prompt = item["document"][0:2000]
        prompts.append(build_prompt_testing(tokenizer, system_prompt, prompt))
    return prompts

def main():

    
    knowledge_dataset = load_from_disk(FOLDER_KNOWLEDGE)
    qa_dataset = load_from_disk(FOLDER_QA)
    
    if TINY:
        print("Using tiny dataset for testing...")
        knowledge_dataset["train"] = knowledge_dataset["train"].select(range(128))
        knowledge_dataset["dev"] = knowledge_dataset["dev"].select(range(64))
        qa_dataset["train"] = qa_dataset["train"].select(range(128))
        qa_dataset["dev"] = qa_dataset["dev"].select(range(64))

    model, tokenizer = get_model()

    if is_adapter:
        import json
        adapter_config_path = os.path.join(MODEL_NAME, "wandb-metadata.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            last_super_epoch = int(adapter_config["config"]["super_epochs"])
            print(f"Loaded adapter train from super epoch {last_super_epoch}")
        else:
            raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}")
    
    # datasets
    qa_dataset_train_text = generate_qa_prompts(qa_dataset["train"], tokenizer)
    qa_dataset_validation_text = generate_qa_prompts(qa_dataset["dev"], tokenizer)
    qa_dataset_test_text = generate_qa_prompts_for_testing(qa_dataset["test"], tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    
    qa_train_dataset = Dataset.from_dict({"text": qa_dataset_train_text})
    qa_val_dataset = Dataset.from_dict({"text": qa_dataset_validation_text})
    qa_test_dataset = Dataset.from_dict({"text": qa_dataset_test_text})

    knowledge_dataset_tokenizer = knowledge_dataset.map(tokenize_function, batched=True)
    qa_train_dataset_tokenizer = qa_train_dataset.map(tokenize_function, batched=True)
    qa_val_dataset_tokenizer = qa_val_dataset.map(tokenize_function, batched=True)
    qa_test_dataset_tokenizer = qa_test_dataset.map(tokenize_function, batched=True)
    
    # training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    auto_config = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        per_device_eval_batch_size=4,        # <-- añade eval batch size
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        save_strategy="no",
        save_total_limit=0,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        # 32 bits
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = SEED,
        report_to = "none", # Use this for WandB etc
        output_dir="./testing/dummy",
    )

    it_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,        # <-- añade eval batch size
        gradient_accumulation_steps=2,
        warmup_steps=25,
        save_strategy="no",
        save_total_limit=0,
        eval_steps=15,
        eval_strategy="steps",         # <-- activa evaluación periódica
        num_train_epochs=1,             # <-- opcional: usa epochs en lugar de max_steps
        #max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim = "paged_adamw_8bit", 
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        output_dir="./testing/dummy",
        load_best_model_at_end=False,          # <-- opcional
        metric_for_best_model="eval_loss",    # <-- opcional
        greater_is_better=False,              # <-- opcional
    )

    trainer_auto = UnslothTrainer(
        model=model,
        train_dataset=knowledge_dataset_tokenizer['train'],
        test_dataset=knowledge_dataset_tokenizer['dev'], # <-- añade validation set
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=auto_config,
    )

    trainer_it = SFTTrainer(
        model=model,
        train_dataset=qa_train_dataset_tokenizer,
        eval_dataset=qa_val_dataset_tokenizer,
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=it_config,
    )

    if not os.path.exists(FOLDER_TO_SAVE_MODELS):
        os.makedirs(FOLDER_TO_SAVE_MODELS)

    if USE_WANDB:
        import wandb
        import json
        wandb.init(
            project=f"llm-recomendation-agent-{dataset_name}", 
            name=f"ccf-{model_basename}-r{RANK_LORA}",
            config={
                "model_name": MODEL_NAME,
                "rank_lora": RANK_LORA,
                "load_in_4bit": LOAD_IN_4BIT,
                "max_length": MAX_LENGTH,
                "super_epochs": SUPER_EPOCHS,
            }
        )

    
    for super_epoch in range(SUPER_EPOCHS):
        if is_adapter: super_epoch += last_super_epoch
        print(f"--- SUPER EPOCH {super_epoch+1} / {SUPER_EPOCHS} ---")
        trainer_sft_stats = trainer_auto.train() 
        trainer_it_stats = trainer_it.train()
        print(f"Super epoch {super_epoch+1} completed.")
        # save adapter
        folder_to_save = os.path.join(name_of_folder_model, f"super_epoch_{super_epoch+1}")
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        model.save_pretrained(folder_to_save)
        print(f"Model saved to {folder_to_save}")
        if USE_WANDB:
            wandb.log({
                "super_epoch": super_epoch + 1,
                "train_loss_sft": trainer_sft_stats.training_loss,
                "train_loss_it": trainer_it_stats.training_loss,
            })
            wandb_info = {
                "run_name": wandb.run.name,
                "project": wandb.run.project,
                "config": dict(wandb.config),
                "url": wandb.run.url
            }
            #save wand info in json in the model folder
            with open(os.path.join(folder_to_save, "wandb-metadata.json"), "w") as f:
                json.dump(wandb_info, f)
            print(f"WandB info saved")

        # Evaluación final en el test set, usar el primero ejemplo del test set para generar una respuesta y compararla con la respuesta real.
        print("Evaluating on test set...")
        test_prompt = qa_dataset_test_text[0]
        test_response = qa_dataset["test"][0]["answer"]
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=MAX_LENGTH)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print(f"Test prompt: {test_prompt}")
            print(f"Generated response: {generated_text}")
            print(f"Expected response: {test_response}")




if __name__ == "__main__":
    #TODO: pasarlo a hydra
    #TODO: tengo que añadir uno script para evaluar los checkpoints con vllm
    #TODO: AÑADIR LA POSIBILEIDA DE EMPEZAR CON UN CHECKPOINT PREVIO
    main()
    