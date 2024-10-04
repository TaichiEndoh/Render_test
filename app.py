import os  # osモジュールをインポートします
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    # 環境変数からPORTを取得し、デフォルトは5000に設定します
    port = int(os.environ.get('PORT', 5000))
    # ホストを「0.0.0.0」にして外部からアクセス可能にします
    app.run(host='0.0.0.0', port=port)