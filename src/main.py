from assets.dataset import DataManager
from assets.model import Model
from assets.train import Train

# データを読み込む（config.py）
data = DataManager()
train_loader, val_loader, classes = data.load(val_ratio=Train.VAL_RATIO)

# モデルの作成
model = Model().build()

# 学習
trainer = Train(model, train_loader, val_loader)
trainer.train()


print("\033[92mAll training completed.\n\033[0m")