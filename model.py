import torch
import torch.nn as nn

class DetermineSpeaker(nn.Module):
    def __init__(self):
        super(DetermineSpeaker, self).__init__()
        self.layer1 = nn.Linear(in_features=4, out_features=16)
        self.layer2 = nn.Linear(in_features=16, out_features=8)
        self.layer3 = nn.Linear(in_features=8, out_features=6)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        return self.layer3(x)