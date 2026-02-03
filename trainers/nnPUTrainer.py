from transformers import Trainer
from trainers.nnPULoss import nnPULoss

class nnPUTrainer(Trainer):
    def __init__(self, *args, prior=0.01, gamma=1.0, beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Instanciamos nuestra loss personalizada
        self.loss_fct = nnPULoss(prior=prior, gamma=gamma, beta=beta)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Aseguramos que labels y logits estén alineados
        loss = self.loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss