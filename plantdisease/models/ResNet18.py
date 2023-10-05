from torch import nn
from torch import flatten
import torch

class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()

        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()

        # BLOCK 1 -> INPUT: 224x224x3, OUTPUT: 112x112x64, s2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)

        # BLOCK 2 (1) -> INPUT: 112x112x64, OUTPUT: 56x56x64
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))

        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 2 (2)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 3 (1) -> INPUT: 56x56x64, OUTPUT: 28x28x128
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_adjust_3 = self.conv4concat(in_ch=64, out_ch=128)
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 3 (2)
        self.conv3_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 4 (1) -> INPUT: 28x28x128, OUTPUT: 14x14x256
        self.conv4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_adjust_4 = self.conv4concat(in_ch=128, out_ch=256)
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 4 (2)
        self.conv4_2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)

        # BLOCK 5 (1) -> INPUT: 14x14x256, OUTPUT: 7x7x512
        self.conv5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_adjust_5 = self.conv4concat(in_ch=256, out_ch=512)
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage) 

        # BLOCK 5 (2)
        self.conv5_2_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)

        # FINAL LAYER
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.fc = nn.Linear(in_features=1*1*512, out_features=1000)
        self.out = nn.Linear(in_features=1000, out_features=n_classes)
            
    
    def conv4concat(self, in_ch, out_ch):
        return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), stride=(2,2), padding=(0,0))      

    def forward(self, x):
        pass





