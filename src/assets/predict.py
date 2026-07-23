import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from assets.dataset import collect_pairs
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

    # 画像と正解ラベルを読み込む
    def load_label(self, label_path):
        label = Image.open(label_path).convert("L")
        label = TF.resize(
            label,
            (config.img_size, config.img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        label = TF.to_tensor(label)
        label = (label > 0.5).to(torch.uint8).squeeze(0).cpu().numpy()

        return label

    # 一枚で推論を行う
    def predict_single(self, image_path):
        image, image_tensor = self.load_image(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob_map = torch.sigmoid(output)[0, 0].cpu().numpy()

        pred_mask = (prob_map > self.threshold).astype(np.uint8)

        return image, prob_map, pred_mask

    # 予測マスクを保存する
    def save_mask(self, pred_mask, save_path):
        mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
        mask_image.save(save_path)

    # 混同行列と各種指標を計算する
    def calculate_metrics(self, pred_mask, label_mask):
        pred = pred_mask.astype(np.uint8)
        label = label_mask.astype(np.uint8)

        tp = int(np.logical_and(pred == 1, label == 1).sum())
        tn = int(np.logical_and(pred == 0, label == 0).sum())
        fp = int(np.logical_and(pred == 1, label == 0).sum())
        fn = int(np.logical_and(pred == 0, label == 1).sum())

        eps = 1e-8
        
        # 
        iou = tp / (tp + fp + fn + eps)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "iou": iou,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy,
        }
        
    # 評価指標をCSVへ保存する
    def save_metrics_csv(self, results, save_dir):

        csv_path = os.path.join(save_dir, "metrics.csv")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Image",
                "Has Crack",
                "TP",
                "TN",
                "FP",
                "FN",
                "IoU",
                "Recall",
                "Precision",
                "F1",
                "Accuracy",
                "GroundTruth Ratio (%)",
                "Prediction Ratio (%)"
            ])

            for result in results:

                metrics = result["metrics"]

                writer.writerow([
                    os.path.basename(result["path"]),
                    "Yes" if result["has_crack"] else "No",
                    metrics["tp"],
                    metrics["tn"],
                    metrics["fp"],
                    metrics["fn"],
                    f"{metrics['iou']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['accuracy']:.6f}",
                    f"{result['gt_ratio']:.4f}",
                    f"{result['pred_ratio']:.4f}",
                ])

        print(f"  Metrics CSV : {csv_path}")
        
    
    # Summaryを整形して表示させる関数
    def print_summary(self, title, summary):

        print()
        print(title)

        for key, value in summary.items():

            if isinstance(value, float):
                print(f"  {key:<10}: {value:.4f}")
            else:
                print(f"  {key:<10}: {value}")
                
    
    # 全体情報を集計する
    def summarize_overall(self, results):
        total = len(results)
        crack = sum(result["has_crack"] for result in results)

        return {
            "Images": total,
            "Crack": crack,
            "Background": total - crack,
        }
    

    # 混同行列を描画して保存する
    def save_confusion_matrix(self, cm, save_path):
        fig, ax = plt.subplots(figsize=(6, 5))

        # 行: 正解, 列: 予測
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Background", "Crack"])
        ax.set_yticklabels(["Background", "Crack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    # 結果を保存する
    def predict_image(self, image_path, label_path, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)

        image, prob_map, pred_mask = self.predict_single(image_path)
        label_mask = self.load_label(label_path)

        fname = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{fname}.png")
        self.save_mask(pred_mask, save_path)

        metrics = self.calculate_metrics(pred_mask, label_mask)
        crack_ratio = pred_mask.mean() * 100
        
        gt_ratio = label_mask.mean() * 100
        has_crack = bool(label_mask.sum())

        print(f"[{os.path.basename(image_path)}]")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        # print(f"  Accuracy: {metrics['accuracy']:.4f}")
        # print(f"    ひび割れ面積率: {crack_ratio:.2f}%")

        return {
            "path": image_path,
            "label_path": label_path,
            "pred_mask": pred_mask,
            
            "has_crack": has_crack,
            
            "gt_ratio": gt_ratio,
            "pred_ratio": crack_ratio,
            
            "metrics": metrics,
        }

    # 複数枚で推論を行う
    def predict_folder(self, img_dir, lab_dir, save_dir="./predict_results"):

        # 画像とラベルのペアを収集する
        pairs = collect_pairs(img_dir, lab_dir)

        if not pairs:
            print(f"Not Bonk!: {img_dir} / {lab_dir}\n")
            return []

        results = []
        total_cm = np.zeros((2, 2), dtype=np.int64)

        for img_path, lab_path in pairs:
            result = self.predict_image(img_path, lab_path, save_dir=save_dir)
            results.append(result)
            total_cm[0, 0] += result["metrics"]["tn"]
            total_cm[0, 1] += result["metrics"]["fp"]
            total_cm[1, 0] += result["metrics"]["fn"]
            total_cm[1, 1] += result["metrics"]["tp"]

        # 集計値を出力する
        tn, fp = total_cm[0, 0], total_cm[0, 1]
        fn, tp = total_cm[1, 0], total_cm[1, 1]
        eps = 1e-8
        summary = {
            "iou": float(tp / (tp + fp + fn + eps)),
            "recall": float(tp / (tp + fn + eps)),
            "precision": float(tp / (tp + fp + eps)),
            "f1": float((2.0 * tp) / (2.0 * tp + fp + fn + eps)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        }

        print("\n[Summary]")
        print(f"  IoU: {summary['iou']:.4f}")
        print(f"  Recall: {summary['recall']:.4f}")
        print(f"  Precision: {summary['precision']:.4f}")
        print(f"  F1: {summary['f1']:.4f}")
        print(f"  Accuracy: {summary['accuracy']:.4f}")

        # 混同行列を result フォルダへ保存する
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        self.save_confusion_matrix(total_cm, cm_path)

        print(f"  Confusion matrix: {cm_path}")
        
        self.save_metrics_csv(results, save_dir)

        return results