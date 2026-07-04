import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,begin_features = 32, num_classes=1000):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, begin_features, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(begin_features)  # Use BatchNorm for the first layer
        self.conv2 = nn.Conv2d(begin_features, begin_features * 2, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(begin_features * 2)  # Use BatchNorm for the second layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv3 = nn.Conv2d(begin_features * 2, begin_features * 4, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(begin_features * 4)  # Use BatchNorm for the third layer
        self.conv4 = nn.Conv2d(begin_features * 4, begin_features * 4, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(begin_features * 4)  # Use BatchNorm for the fourth layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv5 = nn.Conv2d(begin_features * 4, begin_features * 8, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(begin_features * 8)  # Use BatchNorm for the fifth layer
        self.conv6 = nn.Conv2d(begin_features * 8, begin_features * 8, kernel_size=3, padding=1)
        self.norm6 = nn.BatchNorm2d(begin_features * 8)  # Use BatchNorm for the sixth layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(begin_features * 8, 128)
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
    model = CNN(begin_features=1284, num_classes=1000)
    #model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")