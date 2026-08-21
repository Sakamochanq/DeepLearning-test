import segmentation_models_pytorch as smp

from assets.config import config

class Model:
    def build(self):
        
        # セグメンテーションモデルを構築する
        # U-Netを使用
        model = smp.Unet(
            encoder_name = config.encoder_name,
            encoder_weights = config.encoder_weights,
            in_channels = 3,
            
            # ひび割れ 1 / 背景 0
            classes=1
        )
        
        # 既存の学習済みモデルの重みを固定する
        # まず全体を学習する
        for param in model.parameters():
            param.requires_grad = True
            
        # 互換性のある encoder の場合のみ深い層を再学習対象にする
        if hasattr(model.encoder, "layer3"):
            for param in model.encoder.layer3.parameters():
                param.requires_grad = True

        if hasattr(model.encoder, "layer4"):
            for param in model.encoder.layer4.parameters():
                param.requires_grad = True
        
        # デコーダーは常時学習
        for param in model.decoder.parameters():
            param.requires_grad = True
            
        # セグメンテーションヘッド（最終出力層）も常時学習
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        return model