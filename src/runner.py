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


Modes = ["origin", "erosion", "dilation", "opening", "closing"]

print("\n Morphology Mode")


for i, mode in enumerate(Modes, start=1):
    print(f"  {i}. {mode}")

while True:
    try:
        choice = int(input("\n Select ❯ ").strip())

        if 1 <= choice <= len(Modes):
            mode = Modes[choice - 1]
            break

        print(" ？？？.")

    except ValueError:
        print(" Please enter a number.")


print(f"\n Mode : {mode}")



# Predictorの定義
predictor = Predict(model, device, mode=mode)

# DeepCrackのルートディレクトリ（配下に test_img, test_lab を想定）
root = input('\n Root ❯ ').strip()

# 入力と正解ラベルの保存先
img_dir = os.path.join(root, "image")
lab_dir = os.path.join(root, "label")

# 結果の保存先
save_dir = os.path.join(root, "result", mode)

# ひび割れ抽出と評価を実行する
predictor.predict_folder(img_dir, lab_dir, save_dir=save_dir)

print("\033[92m\n Done bonk! ＼(°_o)／ \n\n\033[0m")