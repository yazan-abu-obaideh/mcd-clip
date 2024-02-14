import __main__

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mcd_clip.bike_embedding.ordered_columns import ORDERED_COLUMNS
from mcd_clip.resource_utils import resource_path


class _ResidualBlock(nn.Module):
    def __init__(self, input_size, layer_size, num_layers):
        super(_ResidualBlock, self).__init__()
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


class _ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size, layers_per_block, num_blocks):
        super(_ResidualNetwork, self).__init__()
        self.initial_layer = nn.Linear(input_size, layer_size)
        self.blocks = self._make_blocks(layer_size, layers_per_block, num_blocks)
        self.final_layer = nn.Linear(layer_size, output_size)

    def _make_blocks(self, layer_size, layers_per_block, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_ResidualBlock(layer_size, layer_size, layers_per_block))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.initial_layer(x)
        out = self.blocks(out)
        out = self.final_layer(out)
        return out


__main__.ResidualNetwork = _ResidualNetwork
__main__.ResidualBlock = _ResidualBlock

_DEVICE = torch.device('cpu')
_MODEL_FUNCTION_PATH = resource_path("resnet_0010_0005.pt")
_SCALED_FUNCTION_PATH = resource_path("model_small.pt")
_MODEL_FUNCTION = torch.load(_MODEL_FUNCTION_PATH, map_location=_DEVICE)
_SCALED_FUNCTION = torch.load(_SCALED_FUNCTION_PATH, map_location=_DEVICE)


class EmbeddingPredictor:

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ordered = pd.DataFrame(x, columns=ORDERED_COLUMNS)
        tensor = torch.tensor(ordered.values, dtype=torch.float32)
        result_tensor = _MODEL_FUNCTION(tensor).cpu()
        return result_tensor.detach().numpy()

    def predict_scaled(self, x: pd.DataFrame) -> np.ndarray:
        ordered = pd.DataFrame(x, columns=ORDERED_COLUMNS)
        
        tensor = torch.tensor(ordered.values, dtype=torch.float32)
