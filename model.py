import torch
import torch.nn as nn

class SaunaGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SaunaGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初期隠れ状態
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
