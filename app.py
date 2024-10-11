import os
import torch
import numpy as np
from flask import Flask, request, jsonify

from models.sauna_gru_model import SaunaGRUModel  # 正しいパスに修正

# Flaskアプリの初期化
app = Flask(__name__)

# GRUモデルの読み込み
input_size = 4  # 入力データの次元数（例: 水温、湿度、時間、心拍数など）
hidden_size = 8
output_size = 1  # 整い度のスコア
model = SaunaGRUModel(input_size, hidden_size, output_size)

# モデルの重みを正しいパスでロード
model.load_state_dict(torch.load("./models/sauna_gru_model.pth"))  # パスを修正
model.eval()

# ホームページでAPIの情報とフォームを表示
@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Sauna Condition Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                color: #333;
                margin: 0;
                padding: 20px;
                text-align: center;
            }
            h1 {
                color: #1a73e8;
                font-size: 36px;
            }
            form {
                margin-top: 20px;
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                text-align: left;
            }
            label {
                display: block;
                margin: 10px 0 5px;
                font-size: 18px;
            }
            input[type="number"] {
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .note {
                font-size: 14px;
                color: #555;
                margin-bottom: 10px;
            }
            .btn {
                background-color: #1a73e8;
                color: #fff;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 18px;
            }
            .btn:hover {
                background-color: #155ab6;
            }
            #result {
                margin-top: 20px;
                font-size: 24px;
                font-weight: bold;
            }
            #indicator {
                font-size: 18px;
                margin-top: 10px;
                color: #155ab6;
            }
        </style>
    </head>
    <body>
        <h1>Sauna Condition Prediction API</h1>
        <p>このAPIは、水温、湿度、時間、心拍数などのサウナ条件に基づいて、整い度を予測します。</p>
        <p>This API predicts the relaxation level ("整い度") based on various sauna conditions like water temperature, humidity, time, and heart rate.</p>
        <p>湿度は大体でOKです (Humidity is approximate)</p>
        
        <form id="predict-form">
            <label for="water_temp">Water Temperature (°C):</label><br>
            <input type="number" id="water_temp" name="water_temp" step="0.1" required><br>
            
            <label for="humidity">Humidity (%):</label><br>
            <input type="number" id="humidity" name="humidity" step="0.1" required><br>
            <div class="note">(Humidity is approximate)</div>

            <label for="time_spent">Time Spent (minutes):</label><br>
            <input type="number" id="time_spent" name="time_spent" step="0.1" required><br>
            
            <label for="heart_rate">Heart Rate (bpm):</label><br>
            <input type="number" id="heart_rate" name="heart_rate" step="1" required><br><br>
            
            <input type="button" class="btn" value="Predict" onclick="makePrediction()">
        </form>
        
        <div id="result"></div>
        <div id="indicator"></div>
        
        <script>
            function makePrediction() {
                document.getElementById('result').innerHTML = "Processing...";
                document.getElementById('indicator').innerHTML = "";

                const water_temp = parseFloat(document.getElementById('water_temp').value);
                const humidity = parseFloat(document.getElementById('humidity').value);
                const time_spent = parseFloat(document.getElementById('time_spent').value);
                const heart_rate = parseFloat(document.getElementById('heart_rate').value);
                
                const data = {
                    'features': [
                        [water_temp, humidity, time_spent, heart_rate]
                    ]
                };
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('result').innerHTML = 'Error: ' + data.error;
                    } else {
                        const prediction = data.prediction.toFixed(2);
                        document.getElementById('result').innerHTML = 'Predicted Relaxation Level (整い度): ' + prediction;

                        // 整いやすい指標を追加
                        if (prediction > 70) {
                            document.getElementById('indicator').innerHTML = "整いやすい状態です！ (Optimal for relaxation)";
                        } else if (prediction > 50) {
                            document.getElementById('indicator').innerHTML = "やや整いやすいです (Moderately relaxing)";
                        } else {
                            document.getElementById('indicator').innerHTML = "整うのが難しい状態です (Difficult to relax)";
                        }
                    }
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = 'Error: ' + error;
                });
            }
        </script>
    </body>
    </html>
    '''

# 予測エンドポイント
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
