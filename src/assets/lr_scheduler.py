import math
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, LambdaLR
from assets.config import config


class lr_scheduler:
    
    @staticmethod
    def create(optim):
        if config.type == "StepLR":
            return StepLR(optim, step_size=config.step_size, gamma=config.gamma)
        
        elif config.type == "ExponentialLR":
            return ExponentialLR(optim, gamma=config.gamma)
        
        elif config.type == "CosineAnnealingLR":
            warmup_epochs = getattr(config, 'warmup_epochs', 5)
            total_epochs = config.epochs
            eta_min = getattr(config, 'eta_min', 1e-6)
            base_lr = config.learning_rate

            # どんな環境でも確実に計算通りの学習率にする数式
            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # 1エポック目から5エポック目にかけて、0.1倍から1.0倍に直線的に上昇
                    return 0.1 + 0.9 * (current_epoch / warmup_epochs)
                else:
                    # 後半エポックに向けて滑らかなコサインカーブで減衰
                    progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    progress = min(1.0, max(0.0, progress)) # 1.0を超えないようにガード
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    
                    # 実際の学習率を計算し、初期学習率(base_lr)に対する倍率に変換して返す
                    target_lr = eta_min + (base_lr - eta_min) * cosine_decay
                    return target_lr / base_lr

            # LambdaLR を使えば、SequentialLRのバグを完全に回避できます
            return LambdaLR(optim, lr_lambda)
        
        elif config.type == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optim, mode='min', factor=config.gamma, patience=5)
        
        else:
            print(f"\033[93mlr Scheduler is Bonk!\033[0m")
            return None

    @staticmethod
    def step(scheduler, val_acc=None):
        if isinstance(scheduler, ReduceLROnPlateau):
            if val_acc is None:
                raise ValueError("Val_acc Not Found")
            scheduler.step(val_acc)
        else:
            scheduler.step()
