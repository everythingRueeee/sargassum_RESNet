class Config:
    input_size = (256, 256, 8)  # 7波段 + 1指数共八层
    channels = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_size = 8
    num_epochs = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    pos_weight = 10.0  # 马尾藻样本的损失权重比水样本的损失权重大的倍数
    alpha = 0.25  # 控制类别不平衡的权重，通常设置在 [0.25, 1.0] 之间
    gamma = 2.0   # 控制难分类样本的权重，通常设置为 2
    train_image_dir = 'data/train/images'
    train_mask_dir = 'data/train/masks'
    val_image_dir = 'data/val/images'
    val_mask_dir = 'data/val/masks'
    test_image_dir = 'data/test/images'
    test_mask_dir = 'data/test/masks'
