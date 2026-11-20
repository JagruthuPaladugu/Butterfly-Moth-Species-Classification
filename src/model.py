
import torch
import torch.nn as nn


class ButterflyANN(nn.Module):

    def __init__(self, num_classes=100):
        super(ButterflyANN, self).__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.flatten(x)

        output = self.network(x)

        return  output



