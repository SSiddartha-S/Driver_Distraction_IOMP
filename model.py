# model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        n = x.size(1)
        return x + self.pe[:, :n, :].to(x.device)

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: B x T x D
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x.permute(1,0,2)  # T,B,D
        x = self.transformer(x)
        x = x.mean(dim=0)  # B,D
        return self.classifier(x)
