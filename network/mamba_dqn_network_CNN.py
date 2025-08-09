import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from typing import Tuple

class AtariMambaDQNNetwork(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 sequence_length: int,
                 num_actions: int,
                 d_model: int = 128,
                 d_state: int = 32,
                 d_conv: int = 4,
                 expand: int = 2,
                 fc_hidden_dim1: int = 128,
                 fc_hidden_dim2: int = 128):
        super().__init__()

        self.input_channels = input_shape[0]
        self.num_actions = num_actions
        self.d_model = d_model

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.feature_to_mamba = nn.Linear(cnn_out_dim, d_model)

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fc1 = nn.Linear(d_model, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.q_head = nn.Linear(fc_hidden_dim2, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, C, H, W = x.shape
        x_reshaped = x.view(B * S, C, H, W)
        x_float = x_reshaped.float() / 255.0
        cnn_output = self.cnn(x_float)
        mamba_input = self.feature_to_mamba(cnn_output)
        x_processed = mamba_input.view(B, S, self.d_model)

        mamba_output = self.mamba(x_processed)

        latent_vector = mamba_output[:, -1, :]

        x_head = F.relu(self.fc1(latent_vector))
        x_head = F.relu(self.fc2(x_head))
        q_values = self.q_head(x_head)

        return q_values, latent_vector
