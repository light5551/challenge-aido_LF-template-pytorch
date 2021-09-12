import os
from PIL import Image, ImageDraw, ImageFilter
import torch
# from torch.utils.data import Dataset
import numpy as np
# from tqdm import tqdm
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torch.nn as nn
# from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# Hyperparameters etc.
# LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
PIN_MEMORY = True
NUM_CLASSES = 5

TEST_REAL_DIR = "./real"
LOAD_MODEL_DIR = "/submission/models/"

# !pip3 uninstall albumentations
# !pip3 install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2


# from google.colab import drive
# drive.mount('/content/drive/')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # depthwise conv
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # pointwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=NUM_CLASSES, features=[32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


class Segment:
    test_transform = None
    model = None

    def __init__(self):
        Segment.test_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                ToTensorV2(),
            ],
        )
        Segment.model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
        load_checkpoint(torch.load(f"{LOAD_MODEL_DIR}my_checkpoint1.pth.tar",  map_location=torch.device('cpu')),
                        Segment.model)

    def get_test_transform(self):
        return Segment.test_transform

    def get_model(self):
        return Segment.model
