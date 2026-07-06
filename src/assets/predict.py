import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from assets.config import config


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class Predict:
    def __init__(self, model, device):
        self.device = device
        self.model = model

        # ImageNetの平均と標準偏差で正規化（学習時と合わせる）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # 二値化しきい値
        self.threshold = getattr(config, "threshold", 0.5)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = TF.resize(image, (config.img_size, config.img_size))

        # 画像をテンソル行列に変換
        tensor = TF.to_tensor(image)
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)

        return image, tensor

    # 一枚で推論を行う
    def predict_single(self, image_path):
        image, image_tensor = self.load_image(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob_map = torch.sigmoid(output)[0, 0].cpu().numpy()

        pred_mask = (prob_map > self.threshold).astype(np.uint8)

        return image, prob_map, pred_mask

    # 予測結果をオーバーレイ表示して保存する
    def save_overlay(self, image, pred_mask, save_path):
        image_array = np.array(image).astype(np.float32) / 255.0

        pred_overlay = image_array.copy()
        pred_overlay[pred_mask == 1] = pred_overlay[pred_mask == 1] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image_array)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(pred_overlay)
        axes[1].set_title("Predicted Crack")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    # 結果を保存する
    def predict_image(self, image_path, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)

        image, prob_map, pred_mask = self.predict_single(image_path)

        fname = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{fname}_result.png")
        self.save_overlay(image, pred_mask, save_path)

        crack_ratio = pred_mask.mean() * 100

        print(f"  [{os.path.basename(image_path)}]")
        print(f"    ひび割れ面積率: {crack_ratio:.2f}%")
        print(f"    保存先: {save_path}")

        return {
            "path": image_path,
            "pred_mask": pred_mask,
            "crack_ratio": crack_ratio,
        }

    # 複数枚で推論を行う
    def predict_folder(self, img_dir, save_dir="./predict_results"):
        
        # 画像フォルダ内の画像を取得
        images = [
            os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]

        # 
        if not images:
            print(f"Not found: {img_dir}\n")
            return []

        results = []
        for img_path in images:
            result = self.predict_image(img_path, save_dir=save_dir)
            results.append(result)

        return results