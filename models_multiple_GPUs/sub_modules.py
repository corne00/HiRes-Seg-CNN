import torch
import torch.nn as nn
from .model2d_components import DoubleConv, Down, Up, OutConv

class Encoder(nn.Module):
    """Encoder module for a U-Net architecture."""
    def __init__(self, n_channels=3, depth=4, complexity=32, dropout_rate=0.0):
        """
        Initializes the Encoder module.

        Args:
            n_channels (int): Number of input channels.
            depth (int): Depth of the network (number of blocks).
            complexity (int): Complexity factor controlling the number of channels.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(Encoder, self).__init__()
        
        channels = [complexity * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(n_channels, complexity, dropout_rate=dropout_rate)
        self.contraction = nn.ModuleList()
        
        for i in range(depth):
            self.contraction.append(Down(channels[i], channels[i + 1], dropout_rate=dropout_rate))
        
    def forward(self, x):
        """
        Returns:
            list: List of output tensors at each layer.
        """
        x1 = self.inc(x)
        outputs = [x1]
        for layer in self.contraction:
            x1 = layer(x1)
            outputs.append(x1)
        return outputs
    
class Decoder(nn.Module):
    """Decoder module for a U-Net architecture."""
    def __init__(self, n_channels=3, depth=4, n_classes=3, complexity=32, dropout_rate=0.0):
        """
        Initializes the Decoder module.

        Args:
            n_channels (int): Number of input channels.
            depth (int): Depth of the network.
            n_classes (int): Number of output classes.
            complexity (int): Complexity factor controlling the number of channels.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(Decoder, self).__init__()
        
        channels = [complexity * 2 ** i for i in range(depth + 1)]
        self.expansion = nn.ModuleList()
        
        for i in range(depth):
            self.expansion.append(Up(channels[-1 - i], channels[-2 - i], bilinear=False, dropout_rate=dropout_rate))
        
        self.outc = OutConv(complexity, n_classes)  

    def forward(self, outputs): 
        x2 = outputs[-1]
        for i, layer in enumerate(self.expansion):
            x2 = layer(x2, outputs[-1 - i - 1])
        y = self.outc(x2)
        return y
    
class CNNCommunicator(nn.Module):
    """CNN Communicator module."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, kernel_size=5, padding=2):
        """
        Initializes the CNN Communicator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_rate (float): Dropout rate for regularization.
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (int or tuple): Zero-padding added to both sides of the input.
        """
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        return x