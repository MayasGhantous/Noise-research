import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """A standard block: (Conv2d -> BatchNorm2d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # Using bias=False because BatchNorm handles the bias
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetPreProcessor(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=16):
        super(UNetPreProcessor, self).__init__()
        # --- Encoder (Downsampling) ---
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))
        
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 8, base_features * 16))

        # --- Decoder (Upsampling) ---
        self.up1 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 16, base_features * 8)
        
        self.up2 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 8, base_features * 4)
        
        self.up3 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 4, base_features * 2)
        
        self.up4 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_features * 2, base_features)

        # Final 1x1 convolution to map to output channels
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.inc(x)       # [B, 64, 224, 224]
        x2 = self.down1(x1)    # [B, 128, 112, 112]
        x3 = self.down2(x2)    # [B, 256, 56, 56]
        x4 = self.down3(x3)    # [B, 512, 28, 28]
        x5 = self.down4(x4)    # [B, 1024, 14, 14]

        # Decoder passes with skip connections
        u1 = self.up1(x5)      # [B, 512, 28, 28]
        u1 = torch.cat([x4, u1], dim=1) # Concat channels: 512 + 512 = 1024
        u1 = self.conv_up1(u1) # [B, 512, 28, 28]

        u2 = self.up2(u1)      # [B, 256, 56, 56]
        u2 = torch.cat([x3, u2], dim=1) 
        u2 = self.conv_up2(u2) # [B, 256, 56, 56]

        u3 = self.up3(u2)      # [B, 128, 112, 112]
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv_up3(u3) # [B, 128, 112, 112]

        u4 = self.up4(u3)      # [B, 64, 224, 224]
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv_up4(u4) # [B, 64, 224, 224]

        out = self.outc(u4)    # [B, 3, 224, 224]

        return out
    
class UNetWrapper(nn.Module):
    """
    A wrapper that allows the U-Net to be used as a preprocessor for another model.
    It takes an input image, processes it through the U-Net, and then passes the output
    to the base model (e.g., ViT).
    """
    def __init__(self, base_model, in_channels=3, out_channels=3, base_features=16):
        super(UNetWrapper, self).__init__()
        self.unet = UNetPreProcessor(in_channels=in_channels, out_channels=out_channels, base_features=base_features)
        self.base_model = base_model

    def forward(self, x):
        # Pass input through U-Net
        unet_output = self.unet(x)
        # Pass U-Net output to the base model
        return self.base_model(unet_output)
    
    def get_unet(self):
        return self.unet
    def get_base_model(self):
        return self.base_model
