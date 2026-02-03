import torch
import torch.nn as nn
import torch.nn.functional as F

class nnPULoss(nn.Module):
    def __init__(self, prior, beta=0.0, gamma=1.0):
        """
        Args:
            prior (float): Probabilidad a priori de ser positivo (pi). 
                           Es decir, P(y=1). Debe estimarse del dataset.
            beta (float): Umbral mínimo para el riesgo (normalmente 0).
            gamma (float): Factor de penalización si el riesgo es negativo.
        """
        super().__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 1e-7 # Para evitar divisiones por cero

    def forward(self, logits, targets):
        """
        logits: Salida del modelo sin sigmoide (batch_size, num_labels)
        targets: Etiquetas reales (0 o 1) (batch_size, num_labels)
        """
        # Convertimos targets a float para operaciones matemáticas
        targets = targets.float()
        
        # 1. Definimos los conjuntos P (Positivos) y U (Unlabeled/Negativos)
        # En tu caso, los 1s son P, los 0s son U.
        p_mask = (targets == 1).float()
        u_mask = (targets == 0).float()

        # Contamos n_p (positivos) y n_u (unlabeled)
        n_p = torch.max(p_mask.sum(), torch.tensor(1.0).to(logits.device))
        n_u = torch.max(u_mask.sum(), torch.tensor(1.0).to(logits.device))

        # 2. Calculamos las funciones de pérdida base (Log-Loss / Sigmoid)
        # x = logits
        # loss_pos = -log(sigmoid(x))   --> Lo que queremos para los Positivos
        # loss_neg = -log(1-sigmoid(x)) --> Lo que queremos para los Negativos
        
        # PyTorch BCEWithLogitsLoss ya hace esto numéricamente estable
        # reduction='none' para tener el vector de pérdidas y poder enmascarar
        bce_pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
        bce_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')

        # 3. Estimadores de Riesgo (Risk Estimators)
        
        # Riesgo en Positivos (R_p^+): Media del error en los ejemplos etiquetados como 1
        # (Solo sumamos donde p_mask es 1)
        risk_p = (self.prior / n_p) * (bce_pos * p_mask).sum()

        # Riesgo en Unlabeled tratado como Negativo (R_u^-)
        risk_u_neg = (1.0 / n_u) * (bce_neg * u_mask).sum()

        # Riesgo en Positivos tratado como Negativo (R_p^-) -> El término de corrección
        # "Si supiéramos que son positivos, ¿cuánto error darían si los tratamos como negativos?"
        risk_p_neg = (self.prior / n_p) * (bce_neg * p_mask).sum()

        # 4. Cálculo del Riesgo Negativo Estimado (Unbiased Risk Estimator)
        # R_neg = R_u^- - pi * R_p^-
        estimated_neg_risk = risk_u_neg - risk_p_neg

        # 5. Non-Negative Correction (nnPU)
        # Si estimated_neg_risk < -beta, significa que el modelo está sobreajustando los negativos.
        # En ese caso, la loss es risk_p - beta, y "empujamos" el gradiente con gamma.
        
        if estimated_neg_risk < -self.beta:
            # Caso "Roto": El riesgo negativo es imposible (menor que 0).
            # Ignoramos el riesgo negativo "real" y penalizamos para corregir.
            loss = risk_p - self.beta + (self.gamma * -estimated_neg_risk)
        else:
            # Caso Normal: Sumamos riesgo positivo + riesgo negativo estimado
            loss = risk_p + estimated_neg_risk

        return loss