import torch
import torch.nn as nn

class RegCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(RegCNN, self).__init__() # Regression CNN

        # Each layer: Conv -> Tanh -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ) # out shape: (408, 384)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ) # out shape: (204, 192)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ) # out shape: (102, 96)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ) # out shape: (51, 48)
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ) # out shape: (25, 24)

        final_h = input_height // 32
        final_w = input_width // 32
        
        if final_h == 0 or final_w == 0:
            raise ValueError("Input matrix is too small for 5 layers of pooling!")

        self.fc = nn.Linear(256 * final_h * final_w, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x