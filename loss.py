# coding: utf-8
import torch
import torch.nn as nn

from helpers import getSkeletalModelStructure

# Importar torch_directml para soporte AMD
try:
    import torch_directml

    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False


class Loss(nn.Module):
    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

        # Determinar el device apropiado
        if HAS_DIRECTML:
            self.device = torch_directml.device()
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def forward(self, preds, targets):
        # Asegurar que preds y targets sean float32 para compatibilidad con DirectML
        preds = preds.float()
        targets = targets.float()
        
        # Crear la máscara y asegurar que sea del tipo correcto
        loss_mask = (targets != self.target_pad).float()

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = get_length_direct(targets_masked)

        # Aplicar máscaras con casting explícito
        preds_masked_length = preds_masked_length * loss_mask[:, :, :50]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :50]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :150]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :150]

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked) + 0.1 * self.criterion_bone(
            preds_masked_direct, targets_masked_direct
        )

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

    def cuda(self):
        """Override cuda() method to support DirectML"""
        if HAS_DIRECTML:
            self.to(self.device)
            self.criterion.to(self.device)
            self.criterion_bone.to(self.device)
        else:
            super().cuda()
        return self

    def to(self, device):
        """Override to() method for proper device handling"""
        super().to(device)
        self.criterion.to(device)
        self.criterion_bone.to(device)
        return self


def get_length_direct(trg):
    # Asegurar que trg sea float32
    trg = trg.float()
    
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        # Calcular la diferencia
        diff = trg_list_squeeze[skeleton[0]] - trg_list_squeeze[skeleton[1]]
        
        # Calcular la longitud del esqueleto
        result_length = Skeleton_length = torch.norm(
            diff,
            p=2,
            dim=2,
            keepdim=True,
        )
        
        # Evitar división por cero de manera compatible con diferentes devices
        # Usar un epsilon más grande para DirectML
        epsilon = 1e-8
        result_direct = diff / (Skeleton_length + epsilon)
        
        direct.append(result_direct)
        length.append(result_length)
        
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs