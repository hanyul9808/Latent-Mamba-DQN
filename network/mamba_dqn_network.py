import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from typing import Tuple

class latentMambaDQNNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 sequence_length: int,
                 num_actions: int,
                 mlp_hidden_dim: int = 256,
                 d_model: int = 128,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_actions = num_actions
        self.d_model = d_model

        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.ReLU()
        )
                   
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.output_head = nn.Linear(d_model, num_actions)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, S, current_input_dim = x.shape
        if current_input_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {current_input_dim}.")

        if not torch.is_floating_point(x):
            x = x.float()

        x_reshaped = x.view(B * S, self.input_dim)
        mlp_output = self.input_mlp(x_reshaped)
        x_processed = mlp_output.view(B, S, self.d_model)

        mamba_output = self.mamba(x_processed) # mamba_output shape: (B, S, d_model)

        latent_vector = mamba_output[:, -1, :] # Shape: (B, d_model)

        q_values = self.output_head(latent_vector) # Shape: (B, num_actions)

        return q_values, latent_vector
