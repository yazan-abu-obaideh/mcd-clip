import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, input_size, layer_size, num_layers):
        super(ResidualBlock, self).__init__()
        self.layers = self._make_layers(input_size, layer_size, num_layers)

    def _make_layers(self, input_size, layer_size, num_layers):
        layers = []
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(layer_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        total = out + residual
        return total


# Residual Network class
class ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size, layers_per_block, num_blocks):
        super(ResidualNetwork, self).__init__()
        self.initial_layer = nn.Linear(input_size, layer_size)
        self.blocks = self._make_blocks(layer_size, layers_per_block, num_blocks)
        self.final_layer = nn.Linear(layer_size, output_size)

    def _make_blocks(self, layer_size, layers_per_block, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(layer_size, layer_size, layers_per_block))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.initial_layer(x)
        out = self.blocks(out)
        out = self.final_layer(out)
        return out


device = torch.device('cpu')
model = torch.load("src/resources/resnet_0010_0005.pt", map_location=device)

subset_embeddings = pd.read_csv("test/resources/subset_embeddings.csv", index_col=0)
subset_parametric = pd.read_csv("test/resources/subset_parametric.csv", index_col=0)

x_train, x_test, y_train, y_test = train_test_split(subset_parametric, subset_embeddings, test_size=0.05, random_state=42)
x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.05, random_state=42)
# x_train = torch.tensor(x_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
# y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
# x_train = x_train.to(device)
x_test = x_test.to(device)
# y_train = y_train.to(device)
y_test = y_test.to(device)

targets = y_test.cpu()
preds = model(x_test).cpu()

print(targets)
print(preds)

a_norm = preds / preds.norm(dim=1)[:, None]
b_norm = targets / targets.norm(dim=1)[:, None]
res = torch.mm(a_norm, b_norm.transpose(0, 1))
print(res)
