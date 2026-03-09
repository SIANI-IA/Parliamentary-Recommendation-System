import torch
import torch.nn.functional as F
from transformers import Trainer

class GraphRegularizedTrainer(Trainer):
    def __init__(self, pos_weight_value=None, co_occurrence_matrix=None, lambda_reg=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_value = pos_weight_value
        self.co_occurrence_matrix = co_occurrence_matrix
        self.lambda_reg = lambda_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 1. Pérdida base: BCE con pos_weight (idéntica a tu WeightedTrainer)
        if self.pos_weight_value is not None:
            # Aseguramos que el pos_weight esté en el mismo dispositivo que los logits
            pos_weight = torch.tensor([self.pos_weight_value] * logits.size(-1), device=logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction='mean'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean')

        # 2. Regularización de Grafo (Co-ocurrencia)
        if self.co_occurrence_matrix is not None and self.lambda_reg > 0:
            probs = torch.sigmoid(logits)  # (Batch, M)
            
            # Matriz de diferencias al cuadrado vectorizada: (Batch, M, M)
            diff = probs.unsqueeze(2) - probs.unsqueeze(1)
            squared_diff = diff ** 2
            
            # Aseguramos que S esté en la GPU correcta
            S = self.co_occurrence_matrix.to(logits.device)
            
            # Multiplicamos la topología por las diferencias
            weighted_diff = S.unsqueeze(0) * squared_diff
            
            # Promediamos la penalización del grafo
            reg_loss = weighted_diff.mean()
            
            total_loss = bce_loss + (self.lambda_reg * reg_loss)
        else:
            total_loss = bce_loss

        return (total_loss, outputs) if return_outputs else total_loss