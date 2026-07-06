class config:
    
    # 学習させるデータセット
    dataset = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\DeepCrack";
    
    # 各サブフォルダ名
    train_img_dir = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\DeepCrack\\train_img"
    train_lab_dir = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\DeepCrack\\train_lab"
    test_img_dir = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\DeepCrack\\test_img"
    test_lab_dir = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\DeepCrack\\test_lab"
    
    # 分割シード
    seed = 42

    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 128;
    
    # 学習回数
    epochs = 20;
    
    # 学習率
    learning_rate = 0.0001;
    
    
    # ----- segmentation ----- #
    
    # セグメンテーションモデルのアーキテクチャ
    architecture = "Unet"
    
    # エンコーダー（バックボーン）
    encoder_name = "resnet34"
    
    # エンコーダーの事前学習済み重み（初期値）
    encoder_weights = "imagenet"

    
    # 予測時の二値化しきい値（0.0〜1.0）
    threshold = 0.3
    
    # BCE損失とDice損失の合成比率（1.0でBCEのみ、0.0でDiceのみ）
    bce_weight = 0.2    
    
    # ------------------------ #
 
 
    # 使用する学習モデル
    model = f"Model-{epochs}-{batch_size}-{learning_rate}.pth";
    
    # 学習モデルの保存先
    model_dir = "./";
       
    
    # ----- lr_scheduler ----- #
    
    type = "CosineAnnealingLR"
    step_size = epochs
    gamma = 0.5
    
    # ------------------------ #
    
    