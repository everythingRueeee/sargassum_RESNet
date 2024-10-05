import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset import MyDataset
from models.model import UNetWithResNetBackbone
from config import Config
from sklearn.metrics import f1_score, precision_score, recall_score


def validate_model(model=None, device=None):
    if model is None or device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetWithResNetBackbone(input_channels=Config.input_size[2], output_channels=1).to(device)

        checkpoint = torch.load(r'../weight/best_model.pth', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model from 'weight/best_model.pth'")

    val_dataset = MyDataset(
        image_dir=Config.val_image_dir,
        mask_dir=Config.val_mask_dir,
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    model.eval()  # 设置模型为验证模式
    val_loss = 0

    pos_weight = torch.tensor([Config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            masks = masks.unsqueeze(1)

            outputs = model(images)

            # 记住！！当验证损失使用BCELoss()，它的输入要求是一个概率（范围为 0-1），不能将输出二值化！！BCELoss下面要注释掉
            # outputs = (outputs > 0.5).float()

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # convert outputs to probabilities and then binary predictions
            preds = torch.sigmoid(outputs).data > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)

    # calculate validation F1, precision, recall
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return avg_val_loss

if __name__ == "__main__":
    validate_model()
