import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.sauna_gru_model import SaunaGRUModel  # モデル定義をインポート

# ハイパーパラメータ
input_size = 4  # 入力データの次元数（例: 水温、湿度、時間、心拍数）
hidden_size = 8
output_size = 1  # 整い度のスコア
num_epochs = 1000
learning_rate = 0.001

# ダミーデータ（例: 水温、湿度、時間、心拍数 -> 整い度スコア）
# データサイズ: 100サンプル、各サンプルは4つの特徴量を持つ
np.random.seed(0)
X_train = np.random.rand(100, 10, input_size).astype(np.float32)  # 100サンプル、シーケンス長10、入力次元数4
y_train = np.random.rand(100, output_size).astype(np.float32)  # 出力は1次元

# データをTensorに変換
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

# モデルの初期化
model = SaunaGRUModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# モデルのトレーニング
for epoch in range(num_epochs):
    model.train()  # モデルを訓練モードに設定
    
    # 順伝播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 逆伝播と最適化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルの保存パス
model_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_save_path = os.path.join(model_save_dir, 'sauna_gru_model.pth')
# 保存先ディレクトリが存在しなければ作成
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# モデルの重みを保存
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")
