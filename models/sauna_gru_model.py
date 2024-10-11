import torch
import torch.nn as nn

class SaunaGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SaunaGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.gru.hidden_size)  # 初期の隠れ層の状態
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 最後の時間ステップの出力を使用
        return out

def save_model(model, path='sauna_gru_model.pth'):
    """ モデルの重みをファイルに保存 """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(input_size, hidden_size, output_size, path='sauna_gru_model.pth'):
    """ モデルをロードし、重みを設定 """
    model = SaunaGRUModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(path))
    model.eval()  # 評価モードに設定（ドロップアウトやバッチ正規化を無効化）
    print(f"Model loaded from {path}")
    return model
