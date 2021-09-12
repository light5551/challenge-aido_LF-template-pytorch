import os
from PIL import Image, ImageDraw, ImageFilter
import torch
#from torch.utils.data import Dataset
import numpy as np
#from tqdm import tqdm
import torch.nn as nn
#import torch.optim as optim
#import torchvision
#import torch.nn as nn
#from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# Hyperparameters etc.
#LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
PIN_MEMORY = True
NUM_CLASSES = 5


#!pip3 uninstall albumentations
#!pip3 install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from google.colab import drive
#drive.mount('/content/drive/')



def get_predict(
        image_to_change, model,
        device="cuda"
):
    model.eval()
    x = image_to_change.to(device=device)
    x = torch.unsqueeze(x, 0)
    with torch.no_grad():
        preds = model(x.float())
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=False)
    return preds

def save_preds_image_from_tensor(tenz, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    #------CONVERT "P-mode" tenzor IN IMAGE--------

    #0 - black - [road]
    #1 - white - [roadside]
    #2 - yellow - [markup]
    #3 - red - [crossroads]
    #4 - lilac -[duck]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            #print(tenz.shape)
            elem = tenz[0][i][j]
            if elem == 1:
                image[i][j] = [255, 255, 255]
            elif elem == 2:
                image[i][j] = [255, 255, 0]
            elif elem == 3:
                image[i][j] = [255, 0, 0]
            elif elem == 4:
                image[i][j] = [102, 114, 232]

    return image


def start_segment(image, test_transform, model):
    augmentations = test_transform(image=image)
    image = augmentations["image"]
    preds = get_predict(image, model, device=DEVICE)
    return save_preds_image_from_tensor(preds)

