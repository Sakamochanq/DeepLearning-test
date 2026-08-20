import os
import random

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from assets.config import config


# config から属性を取得する
def get_config_attr(*names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


# 画像ファイルの拡張子
img_exten = {".jpg", ".jpeg", ".png"}


# 画像フォルダとマスク（正解ラベル）フォルダから、ファイル名（拡張子除く）が一致するペアを収集する
def collect_pairs(img_dir: str, lab_dir: str) -> List[Tuple[str, str]]:
    img_files = {p.stem: p for p in Path(img_dir).iterdir() if p.suffix.lower() in img_exten}
    lab_files = {p.stem: p for p in Path(lab_dir).iterdir() if p.suffix.lower() in img_exten}

    pairs = []
    for stem, img_path in sorted(img_files.items()):
        lab_path = lab_files.get(stem)
        if lab_path is None:
            print(f"\033[93mNot Found Mask: {img_path.name}\033[0m")
            continue
        pairs.append((str(img_path), str(lab_path)))

    return pairs

# ひび割れ画像とマスク（正解ラベル）をペアで読み込むDataset
class CrackSegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int, train: bool):
        self.pairs = pairs
        self.img_size = img_size
        self.train = train

        # ImageNetの平均と標準偏差で正規化（画像のみ）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lab_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(lab_path).convert("L")

        # サイズ統一（マスクは最近傍補間でラベル値を保持）
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size),
                          interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            
            # 【変更】±180°の任意回転から、ピクセルが壊れない「90度単位の回転」に変更
            # 0度, 90度, 180度, 270度のいずれかからランダムに選択
            if random.random() < 0.5:
                # 1: 90度, 2: 180度, 3: 270度 回転
                rot_k = random.choice([1, 2, 3])
                image = TF.rotate(image, rot_k * 90)
                mask = TF.rotate(mask, rot_k * 90, interpolation=transforms.InterpolationMode.NEAREST)

            # 50%で水平反転
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 50%で垂直反転（コメントの30%に合わせるなら 0.3 に変更してください）
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 明るさ、コントラスト、彩度をランダムに変化させる（マスクには適用しない）
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = color_jitter(image)
            
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # ランダムに少しぼかす（カーネルサイズは3か5）
                    kernel_size = random.choice([3, 5])
                    image = transforms.GaussianBlur(kernel_size=kernel_size)(image)
                else:
                    # ランダムに輪郭を強調する（シャープネス）
                    sharpness_factor = random.uniform(0.5, 2.0)
                    image = TF.adjust_sharpness(image, sharpness_factor)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = TF.to_tensor(mask)
        
        # マスクを 0.0 / 1.0 の二値へ変換
        mask = (mask > 0.5).float()

        return image, mask

# train と val に分割するためのインデックスを生成する
def train_val_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = int(n * val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    return train_idx, val_idx


class DataManager:
    def __init__(self):
        self.img_size = get_config_attr("IMG_SIZE", "img_size", default=224)

    def load(self, val_ratio: float = 0.15):
        data_root = get_config_attr("DATA_DIR", "dataset")
        if data_root is None:
            raise ValueError("Dataset Bonk!")

        train_img_dir = os.path.join(data_root, get_config_attr("train_img_dir", default="train_img"))
        train_lab_dir = os.path.join(data_root, get_config_attr("train_lab_dir", default="train_lab"))
        # test_img_dir  = os.path.join(data_root, get_config_attr("test_img_dir", default="test_img"))
        # test_lab_dir  = os.path.join(data_root, get_config_attr("test_lab_dir", default="test_lab"))

        # train_img/train_lab のみをまとめて1つのプールにする
        pairs = collect_pairs(train_img_dir, train_lab_dir)
        # pairs += collect_pairs(test_img_dir, test_lab_dir)

        if not pairs:
            raise ValueError(f"Not Found Dataset: {data_root}")

        seed = int(get_config_attr("SEED", "seed", default=42))

        # train_img をまとめて train と val に分割
        train_idx, val_idx = train_val_split_indices(len(pairs), val_ratio=val_ratio, seed=seed)

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]

        # 訓練データはデータ拡張を適用
        train_data = CrackSegDataset(train_pairs, self.img_size, train=True)

        # 検証データは拡張なし
        val_data = CrackSegDataset(val_pairs, self.img_size, train=False)

        batch_size = int(get_config_attr("BATCH_SIZE", "batch_size", default=16))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # セグメンテーションでは classes の代わりに、二値マスクのラベル名を返す
        return train_loader, val_loader, ["Background", "Crack"]
