import torch
import torch.nn as nn
from megadetector.detection.run_detector import load_detector

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetBlock(in_channels, 64, down=True)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True, use_dropout=True)
        self.down5 = UNetBlock(512, 512, down=True, use_dropout=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 256, down=False)
        self.up4 = UNetBlock(512, 128, down=False)
        self.up5 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.bottleneck(d5)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d5], 1))
        u3 = self.up3(torch.cat([u2, d4], 1))
        u4 = self.up4(torch.cat([u3, d3], 1))
        u5 = self.up5(torch.cat([u4, d2], 1))
        out = self.final(torch.cat([u5, d1], 1))
        return out
    

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


class MegaDetectorFeatureExtractor(nn.Module):
    """
    A wrapper for MegaDetector to extract features from an intermediate layer.
    This uses PyTorch hooks to capture the output of a specific layer during the forward pass.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Load the MegaDetector model
        print("Loading MegaDetector for perceptual loss...")
        self.detector = load_detector('MDV5A')
        self.detector.model.eval() # Ensure it's in evaluation mode

        self.features = None
        # The layer name is specific to the YOLOv5 architecture used by MegaDetector.
        # 'model.10' is part of the backbone, capturing mid-level feature for structure-preserving features.
        layer_to_hook = self.detector.model.model[10]
        layer_to_hook.register_forward_hook(self.hook_fn)
        print(f"Hook registered on MegaDetector layer: {layer_to_hook.__class__.__name__}")

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        # run with no_grad to avoid updating the weights
        with torch.no_grad():
            # Rescale from [-1, 1] to [0, 1]
            x_rescaled = (x + 1) / 2.0
            _ = self.detector.model(x_rescaled)
        return self.features