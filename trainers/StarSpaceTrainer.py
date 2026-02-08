from transformers import Trainer
import torch
import torch.nn.functional as F

class StarSpaceTrainer(Trainer):
    def __init__(self, num_negatives=5, **kwargs):
        super().__init__(**kwargs)
        self.num_negatives = num_negatives

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implementación de StarSpace Loss (basada en TED Paper Eq. 1).
        Loss = -log( e^pos / (e^pos + sum(e^neg)) )
        Esto es equivalente a una CrossEntropy donde la clase correcta es la positiva
        y las clases incorrectas son una muestra de K negativos.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        hidden_size = model.config.hidden_size
        logits = outputs.logits / (hidden_size ** 0.5)  # (Batch, Num_Labels) -> Similitud (Dot Product)
        
        total_loss = 0
        valid_samples = 0

        # Iteramos por cada ejemplo en el batch (necesario porque es multi-label)
        for i in range(len(logits)):
            row_logits = logits[i]  # Similitudes para este doc con todos los MPs
            row_labels = labels[i]  # [0, 1, 0, 1...]

            # Identificar índices positivos y negativos
            pos_indices = (row_labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (row_labels == 0).nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0:
                continue # Skip si no hay positivos (raro en train)
            
            valid_samples += 1
            sample_loss = 0

            # StarSpace Strategy: Para CADA positivo, sampleamos K negativos
            for pos_idx in pos_indices:
                # 1. Seleccionar Negativos aleatorios
                if len(neg_indices) > self.num_negatives:
                    # Selección aleatoria sin reemplazo de K negativos
                    perm = torch.randperm(len(neg_indices))[:self.num_negatives]
                    current_neg_indices = neg_indices[perm]
                else:
                    # Si hay menos negativos que K, los usamos todos
                    current_neg_indices = neg_indices

                # 2. Construir el vector de logits para la Softmax
                # El índice 0 será nuestro positivo, el resto los negativos
                selected_indices = torch.cat((pos_idx.unsqueeze(0), current_neg_indices))
                selected_logits = row_logits[selected_indices]

                # 3. Calcular CrossEntropy
                # El target es siempre el índice 0 (que corresponde a pos_idx)
                target = torch.tensor([0], device=logits.device)
                
                # CrossEntropy aplica LogSoftmax internamente: -log(e^x0 / sum(e^xi))
                sample_loss += F.cross_entropy(selected_logits.unsqueeze(0), target)

            # Promediamos la loss sobre el número de positivos en este documento
            total_loss += (sample_loss / len(pos_indices))

        # Promedio sobre el batch
        final_loss = total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (final_loss, outputs) if return_outputs else final_loss