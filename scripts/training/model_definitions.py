import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnModel(nn.Module):
    def __init__(self, input_dim, hidden_size=256, output_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class AttnRnnModel(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_size=256, output_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.seq_length = seq_length

    def forward(self, x):
        x = self.dropout(x)
        out, (h_n, c_n) = self.lstm(x)
        attn_weights = F.softmax(self.attn(h_n[-1]), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), out)
        out = self.fc(attn_applied.squeeze(1))
        return out, attn_weights
