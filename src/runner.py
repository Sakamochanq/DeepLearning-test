import os
import torch

from assets.config import config
from assets.model import Model
from assets.predict import Predict


# モデルの作成
model = Model().build()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(config.model, map_location=device))
model.eval()

# Predictorの定義
predictor = Predict(model, device)

# DeepCrackのルートディレクトリ（配下に test_img, test_lab を想定）
root = input('\n Root ❯ ').strip()

# オーバーレイ画像の保存先
save_dir = os.path.join(root, "predict_results")

# ひび割れ抽出を実行
# test_lab が存在する場合は、正解マスクとの Dice / IoU も計算する
predictor.predict_folder(root, lab_dir=None, save_dir=save_dir)

print("\033[92m\nAll predictions completed.\n\033[0m")
