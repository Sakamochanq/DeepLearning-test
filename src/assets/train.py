import copy

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from assets.config import config
from assets.lr_scheduler import lr_scheduler
from tqdm import tqdm


class DiceBCELoss(nn.Module):
    # BCE損失とDice損失を合成した損失関数。
    # ひび割れは画像全体に対して面積が小さい（クラス不均衡）ため、
    # Diceを混ぜることで少数クラス（ひび割れ）の見逃しを抑える。
    
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)

        probs = torch.sigmoid(outputs)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (union + self.smooth))
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class Train:

    # 検証データの分割比率
    # （train_img + test_img をまとめたデータ全体に対する val の割合。ここを直接編集して調整する）
    VAL_RATIO = 0.15

    # 初期化
    def __init__(self, model, train_loader, val_loader):
        
        # GPUが利用可能であれば使用する。それ以外はCPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用デバイスの確認
        print(f"\033[92m\nUsing device: {self.device}\033[0m")
        
        # GPU/CPUにモデルを送る
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 最適化アルゴリズム ｜ 今回は ADAM を使用
        # 損失関数（BCE + Dice の合成損失）
        self.criterion = DiceBCELoss(bce_weight=getattr(config, "bce_weight", 0.5))
        
        self.optim = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4) # weight_decay は重み減衰。過学習を防ぐための正則化手法（0.0001を適用）
        
        # 学習率スケジューラーの初期化
        self.scheduler = lr_scheduler.create(self.optim)
        
        # 学習曲線用のhistory（accは分類の正解率ではなく Dice係数[%] を記録する）
        self.train_acc_rec = []
        self.val_acc_rec = []
        self.train_loss_rec = []
        self.val_loss_rec = []
        self.lr_rec = []

        # Early Stopping 用の変数
        self._es_counter      = 0            # 改善なし連続エポック数
        self._es_best_loss    = float('inf') # これまでの最小 val_loss
        self._es_best_weights = None         # ベスト時の重み（CPUコピー）

        # 二値化しきい値
        self.threshold = getattr(config, "threshold", 0.5)

    @staticmethod
    def _dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
        
        #バッチ平均のDice係数を計算する
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return dice.mean().item()

    # 学習
    def train(self):
        stopped_early = False  # Early Stopping で中断したかどうか

        for epoch in range(config.epochs):
            # 学習モード ON
            self.model.train()

            # Dice係数の合計
            dice_sum = 0
            
            # バッチ数
            batch_count = 0
            
            # 損失の合計（間違いの合計）
            loss_sum = 0
            
            # 進捗表示
            print('')
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", colour="cyan")
            
            for images, masks in loop:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 勾配降下法の計算を正しく行うため、モデル内のパラメータの勾配を初期化
                # 勾配降下法は、機械学習モデルの誤差（損失）を最小にするパラメータ（重み）を見つけるための反復最適化手法
                # https://www.ibm.com/jp-ja/think/topics/gradient-descent
                self.optim.zero_grad()

                # 予測の出力（1chのロジット画像）
                outputs = self.model(images)
                
                # 出力と正解マスクの損失を計算                
                loss = self.criterion(outputs, masks)
                
                # 勾配を計算
                loss.backward()
                self.optim.step()

                # 損失の加算
                loss_sum = loss_sum + loss.item()

                # 予測確率をしきい値で二値化してDice係数を計算
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    preds = (probs > self.threshold).float()
                    dice = self._dice_score(preds, masks)

                dice_sum += dice
                batch_count += 1
                
                # 更新
                loop.set_postfix(loss=f"{loss_sum / batch_count:.4f}", dice=f"{100 * dice_sum / batch_count:.4f}%")

                # Dice係数の計算（平均）
                train_acc = 100 * dice_sum / batch_count
            
            # 検証のDice係数の計算
            val_acc, val_loss = self.validate()
            
            # 学習曲線用にメモリに記録（エポック全体の平均損失）
            avg_train_loss = loss_sum / len(self.train_loader)
            self.train_acc_rec.append(train_acc)
            self.val_acc_rec.append(val_acc)
            self.train_loss_rec.append(avg_train_loss)
            self.val_loss_rec.append(val_loss)
            
            # 現在の学習率を記録
            current_lr = self.optim.param_groups[0]['lr']
            self.lr_rec.append(current_lr)
            
            # スケジューラーのステップ
            lr_scheduler.step(self.scheduler, val_acc)

            # 結果の出力
            print("\033[96m" + f"学習回数 {epoch+1}/{config.epochs} | " f"訓練損失 {avg_train_loss:.4f} | 検証損失 {val_loss:.4f} | " f"訓練Dice {train_acc:.4f}% | " f"検証Dice {val_acc:.4f}% \n" + "\033[0m")

            # Early Stopping チェック（改善なしが続いたら中断）
            if self.early_stopping(val_loss):
                stopped_early = True
                break

        # 早期終了時はファイル名に earlystop を付加
        suffix = "" if stopped_early else ""

        # 学習モデルの保存
        # sate_dict()でモデルの重みを保存
        model_path = f"{config.model_dir}Model-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.pth"
        torch.save(self.model.state_dict(), model_path)

        #出力する文字を緑にして
        print(f"Model Saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        
        # 学習曲線の可視化
        self.learning_curve(suffix)


    # 検証
    def validate(self):
        
        # 検証モード ON
        self.model.eval()

        dice_sum = 0
        batch_count = 0
        val_loss_sum = 0

        # 勾配を計算しない
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss_sum += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > self.threshold).float()

                dice_sum += self._dice_score(preds, masks)
                batch_count += 1
                
        # 検証のDice係数の計算（平均）
        val_acc = 100 * dice_sum / batch_count
        
        # エポック全体の平均検証損失
        avg_val_loss = val_loss_sum / len(self.val_loader)
        return val_acc, avg_val_loss



    def early_stopping(self, val_loss: float, count_max: int = 5, min_delta: float = 0.0) -> bool:

        # 前回の損失と今回の損失を比較する
        # 前回より損失が低下している場合
        if val_loss < (self._es_best_loss - min_delta):
            
            # 最小損失を更新
            self._es_best_loss    = val_loss
            
            # 損失カウンターをリセット
            self._es_counter = 0
            
            # その時点のモデルの最高精度スコアの重み（パラメータ）をDeepCopyで対応する
            self._es_best_weights = copy.deepcopy(self.model.state_dict())
            
            
            print(f"\033[94m* 損失率 減少 {val_loss:.4f}\033[0m")
            return False

        # 損失が悪化した場合、カウンターを加算
        self._es_counter += 1

        print(f"\033[91m* 損失率 増加 ({self._es_counter}/{count_max})\033[0m")

        # 損失のカウンター最大値に達している場合
        if self._es_counter > count_max:
            
            # 保存しておいた最高精度の重みが存在すれば、モデルに書き戻す
            if self._es_best_weights is not None:
                self.model.load_state_dict(self._es_best_weights)
                
                print(f"\033[91m* Bonk!. (val_loss {self._es_best_loss:.4f})\033[0m\n")
                
            # Trueを返して学習を終了させる
            return True

        # Falseを返して学習を継続させる
        return False



    # 学習曲線の描画
    def learning_curve(self, suffix: str = ""):
        
        epochs = range(1, len(self.train_acc_rec) + 1)
        
        plt.figure(figsize=(16, 5))
        
        # 損失の描画
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_loss_rec, marker='o', label='Train', linewidth=2)
        plt.plot(epochs, self.val_loss_rec, marker='s', label='Valid', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Dice係数の描画
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_acc_rec, marker='o', label='Train', linewidth=2)
        plt.plot(epochs, self.val_acc_rec, marker='s', label='Valid', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score (%)')
        plt.title('Dice Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学習率の推移を描画
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.lr_rec, marker='D', label='Learning Rate', linewidth=2, color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('lr Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
            
        # ./Curve-<epoch>-<batch_size>-<learning_rate>.png として保存 🐧
        pingu = f'Curve-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.png'
        
        plt.savefig(pingu, dpi=300, bbox_inches='tight')
        print(f"{pingu} saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        plt.close()
