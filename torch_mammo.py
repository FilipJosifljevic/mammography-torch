
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv1_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv2_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv2_2.weight, mode='fan_in', nonlinearity='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv3_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv3_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv3_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv4_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv4_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv4_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv5_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv5_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv5_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.model1 = Model1()

        # First convolution block
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='linear')
        self.bn1 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.0)

        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolution block
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='linear')
        self.bn2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.0)

        # Max Pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout3 = nn.Dropout(0.0)

        # Dense Layer
        self.fc = nn.Linear(512, 2)

        self.freeze_except_last_two = False

        self._initialize_requires_grad()

    def _initialize_requires_grad(self):
        # Set requires_grad=False for all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Set requires_grad=True for the last two layers
        for param in self.conv2.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model1(x)

        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Dense Layer
        x = self.fc(x)
        return x
