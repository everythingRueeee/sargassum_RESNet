import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import UNetWithResNetBackbone
from datasets.dataset import MyDataset
from config import Config


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetWithResNetBackbone(input_channels=Config.input_size[2], output_channels=1).to(device)

    checkpoint = torch.load(r'../weight/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print("Loaded model from 'weight/best_model.pth'")

    test_dataset = MyDataset(
        image_dir=Config.test_image_dir,
        mask_dir=Config.test_mask_dir,
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model.eval()
    test_loss = 0

    pos_weight = torch.tensor([Config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            masks = masks.unsqueeze(1)

            outputs = torch.sigmoid(model(images))
            outputs = (outputs > 0.5).float()
            loss = criterion(outputs, masks)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    test_model()
