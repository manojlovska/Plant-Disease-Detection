from torch import nn
from torch import flatten
import torch
from PIL import Image
import torchvision.transforms as transforms

class ResNet9(nn.Module):
    def __init__(self, n_classes):
        super(ResNet9, self).__init__()

        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()

        # BLOCK 1 -> INPUT: 224x224x3, OUTPUT: 224x224x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm1 = nn.BatchNorm2d(64)

        # BLOCK 2 -> INPUT: 224x224x64, OUTPUT: 112x112x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))

        # BLOCK 3 -> INPUT: 112x112x128, OUTPUT: 112x112x128
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1 = nn.BatchNorm2d(128)

        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2 = nn.BatchNorm2d(128)

        # BLOCK 4 -> INPUT: 112x112x256, OUTPUT: 56x56x256
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4 = nn.BatchNorm2d(256)

        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2))

        # BLOCK 5 -> INPUT: 56x56x256, OUTPUT: 28x28x512
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5 = nn.BatchNorm2d(512)

        self.maxpool5 = nn.MaxPool2d(kernel_size=(2,2))

        # BLOCK 6 -> INPUT: 28x28x512, OUTPUT: 28x28x512
        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm6_1 = nn.BatchNorm2d(512)

        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm6_2 = nn.BatchNorm2d(512)

        # FINAL LAYER
        self.maxpool6 = nn.MaxPool2d(kernel_size=(28,28))
        self.flatten = nn.Flatten()
        self.out = nn.Linear(in_features=1024, out_features=n_classes)

    def forward(self, x):
        block1 = self.relu(
            self.batchnorm1(
                self.conv1(x)
            )
        )

        block2 = self.maxpool2(
            self.relu(
                self.batchnorm2(
                    self.conv2(block1)
                )
            )
        )

        block3 = self.relu(
            self.batchnorm3_2(
                self.conv3_2(
                    self.relu(
                        self.batchnorm3_1(
                            self.conv3_1(block2)
                        )
                    )
                )
            )
        )

        block4 = self.maxpool4(
            self.relu(
                self.batchnorm4(
                    self.conv4(torch.cat((block2, block3), axis=1))
                )
            )
        )
        
        block5 = self.maxpool5(
            self.relu(
                self.batchnorm5(
                    self.conv5(block4)
                )
            )
        )

        block6 = self.relu(
            self.batchnorm6_2(
                self.conv6_2(
                    self.relu(
                        self.batchnorm6_1(
                            self.conv6_1(block5)
                        )
                    )
                )
            )
        )

        output = self.out(
            self.flatten(
                self.maxpool6(torch.cat((block5, block6), axis=1))
            )
        )

        return output


# model = ResNet9(n_classes=10)

# image = Image.open('/home/anastasija/Documents/IJS/Projects/PlantDiseaseDetection/Plant-Disease-Detection/datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/train/Apple___Apple_scab/image (1).JPG')
  
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor()
# ])
  
# # transform = transforms.PILToTensor()
# # Convert the PIL image to Torch tensor
# img_tensor = transform(image)
# # print(img_tensor.shape)
# img_tensor = torch.unsqueeze(img_tensor, dim=0)
# print(img_tensor.shape)
# # print(img_tensor)


# tensor = model(img_tensor)
# print(tensor.shape)
# print(tensor)

