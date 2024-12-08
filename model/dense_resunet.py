import torch
import torch.nn as nn
import torch.nn.functional as F

# T-Max-Avg Pooling Layer
class TMaxAvgPooling(nn.Module):
    def __init__(self, T=0.5):
        super(TMaxAvgPooling, self).__init__()
        self.T = T

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return self.T * max_pool + (1 - self.T) * avg_pool

# Dense ResU-Net Model
class DenseResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DenseResUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256 + 256, 128)
        self.upconv1 = self.upconv_block(128 + 128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64 + 64, 128, kernel_size=1)  # 128 channels after concatenation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(128, 1)  # Fully connected layer with 128 input features

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return upconv

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        # Decoder path
        up3 = self.upconv3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)

        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2], dim=1)z

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], dim=1)

        # Output layer
        output = self.output_conv(up1)
        output = self.global_pool(output)
        output = output.view(output.size(0), -1)  # Flatten
        output = self.fc(output)
        return output
