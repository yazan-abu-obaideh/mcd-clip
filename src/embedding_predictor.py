import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import __main__


class ResidualBlock(nn.Module):
    def __init__(self, input_size, layer_size, num_layers):
        super(ResidualBlock, self).__init__()
        self.layers = self._make_layers(input_size, layer_size, num_layers)

    def _make_layers(self, input_size, layer_size, num_layers):
        layers = [nn.Linear(input_size, layer_size), nn.ReLU()]
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


__main__.ResidualNetwork = ResidualNetwork
__main__.ResidualBlock = ResidualBlock

_DEVICE = torch.device('cpu')
_MODEL_FUNCTION_PATH = os.path.join(os.path.dirname(__file__), "resources", "resnet_0010_0005.pt")
_MODEL_FUNCTION = torch.load(_MODEL_FUNCTION_PATH, map_location=_DEVICE)


class EmbeddingPredictor:

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        tensor = torch.tensor(x.values, dtype=torch.float32)
        result_tensor = _MODEL_FUNCTION(tensor).cpu()
        return result_tensor.detach().numpy()
