import torch
import torch.nn as nn
import torch.optim as optim
from spoter import spoter_model

class LinearSpoter(nn.Module):

    def __init__(self, prev_num_classes):
        super().__init__()

        # load the previous model of x classes
        self.pretrained_model = spoter_model.SPOTER(prev_num_classes, hidden_dim=54*2)

        # add aditional linear layer
        self.additional_layers = nn.ModuleList()  # list of linear layers

        for _ in range(int(prev_num_classes/10)-1):
            self.additional_layers.append(nn.Linear(self.pretrained_model.hidden_dim, 1))

    def forward(self, inputs):

        out = self.pretrained_model(inputs)
        out = self.additional_layers(out)
        return out

    def add_layer(self):
        # Add an additional linear layer
        self.additional_layers.append(nn.Linear(self.pretrained_model.hidden_dim, 1))