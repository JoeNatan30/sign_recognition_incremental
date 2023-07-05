import torch
import torch.nn as nn
import torch.optim as optim
from spoter import spoter_model

class simpleSpoter(nn.Module):

    def __init__(self, prev_num_classes):
        super().__init__()

        # Cargar el modelo previamente entrenado para x clases
        self.pretrained_model = spoter_model.SPOTER(prev_num_classes, hidden_dim=54*2)

    def forward(self, inputs):

        return self.pretrained_model(inputs)