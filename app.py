import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from model import SaunaGRUModel

# Flaskアプリの初期化
app = Flask(__name__)

# GRUモデルの読み込み
input_size = 4  # 入力データの次元数（例: 水温、湿度、時間、心拍数など）
hidden_size = 8
output_size = 1  # 整い度のスコア
model = SaunaGRUModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("sauna_gru_model.pth"))  # モデルの重みをロード
model.eval()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # POSTされたデータを取得
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'No input data provided'}), 400

    # 特徴量を読み込み、Tensorに変換
    features = np.array(data['features']).astype(np.float32)
    features = torch.tensor(features).view(1, -1, input_size)

    # モデルによる予測
    with torch.no_grad():
        prediction = model(features)
    
    # 予測結果を返す
    return jsonify({'prediction': prediction.item()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
