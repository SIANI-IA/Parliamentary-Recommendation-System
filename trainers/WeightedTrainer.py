from transformers import Trainer
import torch
import torch.nn as nn

class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight_value=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Registramos el peso como un tensor en el dispositivo correcto
        self.pos_weight = torch.tensor([pos_weight_value]).to(self.args.device) if self.args.device else torch.tensor([pos_weight_value])
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Mover el peso al mismo dispositivo que los logits si es necesario
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        # Definimos la función con el peso para los positivos
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss