import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.model import UNetWithResNetBackbone
from datasets.dataset import MyDataset
from config import Config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def predict_image(model, image, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Adjust the image dimensions
        image = image.squeeze(1).to(device)
        output = model(image.to(device))
        output = torch.sigmoid(output)
        return (output > 0.5).float()


def plot_prediction(image, mask, prediction):
    fig, axs = plt.subplots(Config.batch_size, Config.input_size[2], figsize=(15, 15))  # 行（对应 batch size），列（对应通道数）

    for i in range(Config.batch_size):  # 遍历 batch size
        for j in range(Config.input_size[2]):  # 遍历通道
            ax = axs[i, j]  # 每个批次的每个通道对应一个子图
            ax.imshow(image[i, j, :, :], cmap='gray')
            ax.set_title(f'Batch {i + 1}, Channel {j + 1}')
            ax.axis('off')  # 可选：隐藏坐标轴

    plt.tight_layout()
    plt.show()

    # 另外显示mask和预测结果
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    mask_to_display = mask[0].squeeze()
    prediction_to_display = prediction[0].squeeze().cpu().numpy()

    axs[0].imshow(mask_to_display, cmap='gray')
    axs[0].set_title('Ground Truth Mask')

    axs[1].imshow(prediction_to_display, cmap='gray')
    axs[1].set_title('Prediction')

    plt.show()


def main():
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetWithResNetBackbone(input_channels=Config.input_size[2], output_channels=1).to(device)

    checkpoint = torch.load(r'../weight/epoc100/best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    pre_dataset = MyDataset(
        image_dir=r'../data/train/images',
        mask_dir=r'../data/train/masks',
        is_train=False
    )
    pre_loader = DataLoader(pre_dataset, batch_size=Config.batch_size, shuffle=False)

    # 进行预测
    for i, (image, mask) in enumerate(pre_loader):
        image = image.to(device)
        prediction = predict_image(model, image, device)

        plot_prediction(image.cpu().numpy(), mask.squeeze().numpy(), prediction)

if __name__ == "__main__":
    main()
