import torch
import torch.nn as nn
import torchvision.models as models

class UNetWithResNetBackbone(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetWithResNetBackbone, self).__init__()

        # Use ResNet as the encoder
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify the first convolution layer to accommodate input_channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract the layers of ResNet for use as encoder
        self.resnet_layers = list(self.resnet.children())
        self.encoder1 = nn.Sequential(*self.resnet_layers[:3])  # Conv1 and relu, maxpool
        self.encoder2 = nn.Sequential(*self.resnet_layers[3:5])  # res Layer1
        self.encoder3 = nn.Sequential(self.resnet_layers[5])  # res Layer2
        self.encoder4 = nn.Sequential(self.resnet_layers[6])  # res Layer3
        self.encoder5 = nn.Sequential(self.resnet_layers[7])  # res Layer4

        self.dropout = nn.Dropout(0.5)  # dropout with 50%

        # Decoder layers with transpose convolutions to upsample
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 8x8 to 16x16, channel 512 to 256
        self.up_conv5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 -> 256 (concat 256+256)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 16x16 to 32x32
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 to 128 (concat 128+128)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 32x32 to 64x64
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 to 64 (concat 64+64)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64x64 to 128x128
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 96 to 32 (concat 64+32)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.final_up = nn.ConvTranspose2d(32, output_channels, kernel_size=2, stride=2)  # 128x128 to 256x256

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.dropout(e1))
        e3 = self.encoder3(self.dropout(e2))
        e4 = self.encoder4(self.dropout(e3))
        e5 = self.encoder5(self.dropout(e4))

        # Decoding path
        u5 = self.up5(e5)
        u5 = torch.cat([u5, e4], dim=1)  # Concatenate with e4, encoder + decoder
        u5 = self.up_conv5(u5)

        u4 = self.up4(u5)
        u4 = torch.cat([u4, e3], dim=1)  # Concatenate with e3
        u4 = self.up_conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, e2], dim=1)  # Concatenate with e2
        u3 = self.up_conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, e1], dim=1)  # Concatenate with e1
        u2 = self.up_conv2(u2)

        output = self.final_up(u2)  # Final upsample to get 256x256 output

        return output

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Add Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Add Batch Normalization
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(input_channels, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        self.down5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv5 = conv_block(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

        self.apply(self.init_weights)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.pool(d1)
        d2 = self.down2(d2)
        d3 = self.pool(d2)
        d3 = self.down3(d3)
        d4 = self.pool(d3)
        d4 = self.down4(d4)
        d5 = self.pool(d4)
        d5 = self.dropout(self.down5(d5))

        u5 = self.up5(d5)
        u5 = torch.cat([u5, d4], dim=1)
        u5 = self.up_conv5(u5)

        u4 = self.up4(u5)
        u4 = torch.cat([u4, d3], dim=1)
        u4 = self.up_conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.up_conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_conv2(u2)

        output = self.final_conv(u2)
        return output

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
