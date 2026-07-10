import segmentation_models_pytorch as smp

from assets.config import config

class Model:
    def build(self):
        
        # 既存の学習済みモデル ResNet18 を使用する
        model = smp.Unet(
            encoder_name = config.encoder_name,
            encoder_weights = config.encoder_weights,
            in_channels = 3,
            
            # ひび割れ 1 / 背景 0
            classes=1
        )
        
        # 既存の学習済みモデルの重みを固定する
        # つまり、ResNet18の特徴抽出部分は学習させず、最終層のみを学習させる
        for param in model.parameters():
            param.requires_grad = True
            
        # レイヤー1層目の追加
        for param in model.encoder.layer1.parameters():
            param.requires_grad = True
            
        # レイヤー3層目の追加
        for param in model.encoder.layer3.parameters():
            param.requires_grad = True
        
        # レイヤー4層目の追加
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True
        
        # デコーダーは常時学習
        for param in model.decoder.parameters():
            param.requires_grad = True
            
        # セグメンテーションヘッド（最終出力層）も常時学習
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        return model