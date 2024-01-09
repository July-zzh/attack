import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    def __init__(self):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),  # 14 * 14
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),

            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 28 * 28
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 56 * 56
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 112 * 112
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 224 * 224
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 2048, 7, 7)
        out = self.conv(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution layer 1  n = (（w - f + 2 * p）/ s ) + 1计算公式 N是输出尺寸，W是输入尺寸，F是卷积核或池化窗的尺寸，P是填充值的大小，S是步长大小。
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)  # [32,220,220]
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)  # [32,216,216]
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size: [32,108,108]
        self.conv1_drop = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)  # [64,106,106]
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)  # [64,104,104]
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size: [64,52,52]
        self.conv2_drop = nn.Dropout(0.25)

        # Convolution layer 3
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                               padding=0)  # output size: [128,50,50]
        self.relu5 = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                               padding=0)  # output size: [128,48,48]
        self.relu6 = nn.ReLU()
        self.batch6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size: [128,24,24]
        self.conv3_drop = nn.Dropout(0.25)

        # Convolution layer 4
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                               padding=0)  # output size: [256,22,22]
        self.relu7 = nn.ReLU()
        self.batch7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                               padding=0)  # output size: [256,20,20]
        self.relu8 = nn.ReLU()
        self.batch8 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size: [256,10,10]
        self.conv4_drop = nn.Dropout(0.25)

        # Convolution layer 5
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                               padding=0)  # output size: [512,8,8]
        self.relu9 = nn.ReLU()
        self.batch9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                padding=0)  # output size: [512,6,6]
        self.relu10 = nn.ReLU()
        self.batch10 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size: [512,3,3]
        self.conv5_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1

        self.fc1 = nn.Linear(4608, 1024)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2 的前向计算，4行代码
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)

        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        # conv layer 3 的前向计算，4行代码
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.batch5(out)

        out = self.conv6(out)
        out = self.relu6(out)
        out = self.batch6(out)

        out = self.maxpool3(out)
        out = self.conv3_drop(out)

        # conv layer 4 的前向计算，4行代码
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.batch7(out)

        out = self.conv8(out)
        out = self.relu8(out)
        out = self.batch8(out)

        out = self.maxpool4(out)
        out = self.conv4_drop(out)

        # conv layer 5 的前向计算，4行代码
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.batch9(out)

        out = self.conv10(out)
        out = self.relu10(out)
        out = self.batch10(out)

        out = self.maxpool5(out)
        out = self.conv5_drop(out)

        # Flatten拉平操作
        out = out.view(out.size(0), -1)  # 4608

        # FC layer的前向计算（2行代码）
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)

        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) # output size: [16,220,220]
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output size: [16,110,110]

        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # output size: [32,106,106]
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output size: [32,53,53]

        # Fully-Connected layer 1
        self.fc1 = nn.Linear(89888, 256) # input size: 32*53*53=89888
        self.fc1_relu = nn.ReLU()

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1 的前向计算，2行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # conv layer 2 的前向计算，2行代码
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Flatten拉平操作
        out = out.view(out.size(0), -1) # output size: 89888

        # FC layer的前向计算（2行代码）
        out = self.fc1(out) # output size: 256
        out = self.fc1_relu(out)

        out = self.fc2(out) # output size: 10

        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    model1 = Generator()
    model2 = CNN()
    model = nn.Sequential(model1, model2)
