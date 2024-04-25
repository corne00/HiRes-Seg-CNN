import torch
import torch.nn as nn
import torch.nn.functional as F

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CNNCommunicator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, kernel_size=3, padding=1):
        super(CNNCommunicator, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # Second convolutional layer
        self.conv2 = nn.Conv2d((in_channels + out_channels) // 2, (in_channels + out_channels) // 2, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d((in_channels + out_channels) // 2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Third convolutional layer
        self.conv3 = nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        #print(x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        return x

class UpResNet(nn.Module):
    """
    UpResNet module for image processing tasks.

    Args:
        in_channels_1 (int): Number of input channels for the first input.
        in_channels_2 (int): Number of input channels for the second input.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): Whether to use bilinear interpolation. Default is True.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpResNet, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1 // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1 // 2 + in_channels_2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        """
        Forward pass for the UpResNet module.

        Args:
            x1 (torch.Tensor): Input tensor from the first pathway.
            x2 (torch.Tensor): Input tensor from the second pathway.

        Returns:
            torch.Tensor: Output tensor.
        """

        x1 = self.up(x1)

        # Pad x1 to have the same size as x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class UpComm(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpComm, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(2 * in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCommunicator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, bilinear=False):
        super(UNetCommunicator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, in_channels))
        self.down1 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))
        self.down2 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))
        self.down3 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))

        factor = 2 if bilinear else 1
        
        self.up1 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.up2 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.up3 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.outc = (OutConv(in_channels, out_channels))

    def forward(self, data):
        x1 = self.inc(data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        output = self.outc(x)

        return output