import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import UNetWithResNetBackbone
from datasets.dataset import MyDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scripts.validate import validate_model
from config import Config
from sklearn.metrics import f1_score, precision_score, recall_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetWithResNetBackbone(input_channels=Config.input_size[2], output_channels=1).to(device)
    model.apply(model.init_weights)  # initialize weights

    # calculate pos_weight for unbalanced data
    pos_weight = torch.tensor([Config.pos_weight]).to(device)  # Use pos_weight to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    # criterion = FocalLoss(alpha=Config.alpha, gamma=Config.gamma).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    train_dataset = MyDataset(
        image_dir=Config.train_image_dir,
        mask_dir=Config.train_mask_dir,
        is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

    val_losses = []
    train_losses = []
    f1_scores = []
    precisions = []
    recalls = []

    best_val_loss = float('inf')
    for epoch in range(Config.num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device).float()
            masks = masks.unsqueeze(1)  # add channel dimension

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()  # computes the gradient of the loss
            optimizer.step()  # updating the model's weights

            epoch_loss += loss.item()

            # convert outputs to probabilities and then binary predictions
            preds = torch.sigmoid(outputs).data > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

        # calculate F1, precision, recall at the end of the epoch
        f1 = f1_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{Config.num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # validate after each epoch
        val_loss = validate_model(model, device)  # call the validate_model() to validate
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Current learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), r'weight/best_model.pth')

    torch.save(model.state_dict(), r'weight/last_model.pth')

    # plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Config.num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, Config.num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.show()

    # plot F1-score, Precision, Recall
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Config.num_epochs + 1), f1_scores, label='F1 Score')
    plt.plot(range(1, Config.num_epochs + 1), precisions, label='Precision')
    plt.plot(range(1, Config.num_epochs + 1), recalls, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 Score, Precision, and Recall')
    plt.legend()
    plt.savefig('metrics.png')
    plt.show()


if __name__ == "__main__":
    train_model()
