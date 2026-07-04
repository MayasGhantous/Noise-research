import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from archtechre_common import *
from Unet import UNetWrapper

class CNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(16, 128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(16, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm5 = nn.GroupNorm(32, 256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm6 = nn.GroupNorm(32, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.norm_fc = nn.LayerNorm(128)
        self.drop = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.norm6(self.conv6(x)))
        x = self.pool3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.norm_fc(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)

        return x

    def initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = UNetWrapper(CNN(begin_features=128, num_classes=1000))
    #model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = CNN(begin_features=128, num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")