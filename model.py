# model.py
import torch
import torch.nn as nn

NUM_OPS = 8
INPUT_DIM = 20

class OPiPredictor(nn.Module):
    def __init__(self):
        super(OPiPredictor, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM, 64, num_layers=2, bidirectional=True, batch_first=True)
        self.attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(128)  # ✅ added normalization
        self.dropout = nn.Dropout(0.2)  # ✅ added dropout

        self.exist_fc = nn.Linear(128, NUM_OPS)
        self.count_fc = nn.Linear(128, NUM_OPS)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(attn_out)
        attn_out = self.dropout(attn_out)

        exist_pred = torch.sigmoid(self.exist_fc(attn_out))
        count_pred = torch.relu(self.count_fc(attn_out))
        count_pred = (count_pred - count_pred.mean()) / (count_pred.std() + 1e-5)

        fused = exist_pred + count_pred
        return exist_pred, count_pred, fused


class LayerDecoder(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, 128, num_layers=2, bidirectional=True, batch_first=True)

        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.15)
        self.gelu = nn.GELU()
        self.decoder = nn.Linear(256, vocab_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        nn.init.constant_(self.decoder.bias, 0.01)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        enc_out = self.norm(enc_out)
        enc_out = self.dropout(self.gelu(enc_out))
        return self.decoder(enc_out)