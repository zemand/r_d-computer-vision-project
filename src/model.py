"""Маленький U-Net: denoising convolutional encoder-decoder для реставрації кадру.

Encoder стискає кадр (бачить велику картину: відблиск, нерівне світло),
decoder розгортає назад, а skip connections передають дрібні деталі (тонкі смуги)
напряму з encoder у decoder, щоб вони не розмились.
"""

import cv2
import numpy as np
import torch
from torch import nn

DEPTH = 4                 # скільки разів зменшуємо кадр удвічі
MULTIPLE = 2 ** DEPTH     # тому розмір кадру має ділитись на 16


def conv_block(in_channels, out_channels):
    """Два згорткові шари 3×3 + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        channels = [base * 2 ** i for i in range(DEPTH + 1)]   # 16, 32, 64, 128, 256
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_channels = 1
        for out_channels in channels[:-1]:
            self.down.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.bottom = conv_block(channels[-2], channels[-1])
        for i in range(DEPTH, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2))
            # після склеювання зі skip connection каналів удвічі більше
            self.up_blocks.append(conv_block(channels[i - 1] * 2, channels[i - 1]))
        self.out = nn.Conv2d(channels[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)           # запам'ятовуємо деталі для decoder
            x = self.pool(x)
        x = self.bottom(x)
        for up, block, skip in zip(self.up, self.up_blocks, reversed(skips)):
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.out(x))   # яскравість 0..1


def get_device():
    """GPU Mac (MPS), якщо є, інакше CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path, device=None):
    if device is None:
        device = get_device()
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def restore(model, image):
    """Сірий кадр uint8 будь-якого розміру -> відреставрований кадр uint8 того ж розміру."""
    height, width = image.shape
    # доповнюємо до кратного 16 повтором країв, потім обрізаємо назад
    pad_bottom = (-height) % MULTIPLE
    pad_right = (-width) % MULTIPLE
    padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0)[None, None].to(device)
    output = model(tensor)[0, 0, :height, :width].cpu().numpy()
    return (output * 255).round().astype(np.uint8)
