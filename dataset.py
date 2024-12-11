import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

        # 计算每个通道的均值和标准差
        self.mean, self.std = self.calculate_mean_std()

        if self.is_train:
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: self.normalize_per_channel(img))  # 使用自定义标准化
            ])
        else:
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: self.normalize_per_channel(img))  # 使用自定义标准化
            ])

    def __len__(self):
        # 返回数据集中图像的数量
        return len(self.images)

    def calculate_mean_std(self):
        num_channels = Config.input_size[2]
        pixel_sum = np.zeros(num_channels)
        pixel_squared_sum = np.zeros(num_channels)
        num_pixels = np.zeros(num_channels)

        for image_name in self.images:
            img_path = os.path.join(self.image_dir, image_name)
            with rasterio.open(img_path) as src:
                image = src.read().astype(np.float32)
                image = image[Config.channels]

            # 遍历每个通道计算
            for i in range(image.shape[0]):
                channel_data = image[i, :, :]
                pixel_sum[i] += np.nansum(channel_data)  # 使用 np.nansum 来忽略 NaN
                pixel_squared_sum[i] += np.nansum(channel_data ** 2)  # 使用 np.nansum 来忽略 NaN
                num_pixels[i] += np.sum(~np.isnan(channel_data))  # 仅计算非 NaN 像素数量

        # 计算每个通道的均值和标准差
        mean = pixel_sum / num_pixels
        std = np.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))
        std[std == 0] = 1e-8  # 防止除 0 错误

        # 测试 mean std 合理性，可注掉
        # print(f"Calculated mean: {mean}, std: {std}")

        return mean.tolist(), std.tolist()

    def normalize_per_channel(self, img):
        # 按通道标准化
        normalized_channels = []
        for i in range(img.shape[0]):
            normalized_channel = (img[i, :, :] - self.mean[i]) / (self.std[i] if self.std[i] != 0 else 1e-8)  # 防止除 0 错误
            normalized_channels.append(normalized_channel)
        return torch.stack(normalized_channels, dim=0)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("_img.tif", "_msk.tif"))

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = image[Config.channels]

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # 对每个通道进行增强并堆叠成多通道图像
        channels = [image[i, :, :] for i in range(image.shape[0])]
        augmented_channels = [self.augmentation(channel).squeeze(0) for channel in channels]

        # 合并通道，确保输入形状为 [num_channels, height, width]
        image_tensor = torch.stack(augmented_channels, dim=0)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)
            mask = np.nan_to_num(mask, nan=0.0)  # 将 NaN 值替换为 0.0

        mask_tensor = torch.tensor(mask).float()

        return image_tensor, mask_tensor
