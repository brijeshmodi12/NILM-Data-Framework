import torch
import torch.nn as nn
import numpy as np

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1024, d_model=128, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].to(x.device))

class BERT4NILM(nn.Module):
    def __init__(self, seq_len, d_model=128, n_head=8, n_layers=4, d_feedforward=256, dropout=0.1, cnn_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=7, padding=3)
        self.cnn_activation = nn.ReLU()
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(max_len=seq_len, d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)           # [B, 1, L]
        x = self.conv1(x)            # [B, C, L]
        x = self.cnn_activation(x)
        x = x.permute(0, 2, 1)       # [B, L, C]
        x = self.embedding(x)        # [B, L, d_model]
        x = self.input_norm(x)
        x = self.dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x_decoded = self.decoder(x)           # [B, L, 1]
        return x_decoded.squeeze(-1)  