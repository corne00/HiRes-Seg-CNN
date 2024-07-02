import torch
import torch.nn as nn
import copy
import torchvision.models
from models_2d.model2d_components import DoubleConv,CNNCommunicator, OutConv
import torch.nn.functional as F

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
    
class OutResNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(OutResNet, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2, in_channels // 2, dropout_rate=dropout_rate)
        self.outc = OutConv(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return self.outc(x)

    
class ResNetEncoder(nn.Module):
    """Encoder module for a ResNetU-Net architecture."""
    def __init__(self, resnet_type = "resnet18", extra_conv = True, n_channels=3, depth=4, complexity=32, dropout_rate=0.0):
        super(ResNetEncoder, self).__init__()
        
        self.resnet_type = resnet_type
        self.extra_conv = extra_conv
        
        if self.resnet_type == "resnet18":
            base_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        elif self.resnet_type == "resnet34":
            base_model = torchvision.models.resnet34(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        else:
            base_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]

        self.encoder_layers = list(base_model.children())
        
        self.block1 = nn.Sequential(*self.encoder_layers[:3])  # 3   -->   64
        self.block2 = nn.Sequential(*self.encoder_layers[3:5]) # 64  -->   64 
        self.block3 = nn.Sequential(*self.encoder_layers[5])   # 64  -->  128  
        self.block4 = nn.Sequential(*self.encoder_layers[6])   # 128 -->  256 
        
        if self.extra_conv:
            self.conv = DoubleConv(self.channel_distribution[-1], self.channel_distribution[-1], dropout_rate=dropout_rate)
            
    def forward(self, x):
        """
        Returns:
            list: List of output tensors at each layer.
        """
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        if self.extra_conv:
            x4 = self.conv(self.block4(x3))
        else:
            x4 = self.block4(x3)
            
        outputs = [x1, x2, x3, x4]
        return outputs
    
    
class ResNetDecoder(nn.Module):
    """Decoder module for a U-Net architecture."""
    def __init__(self, n_channels=3, n_classes=3, dropout_rate=0.0, bilinear=False):
        super(ResNetDecoder, self).__init__()
        
        channels = [3, 64, 64, 128, 256]
        self.expansion = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        self.up1 = UpResNet(256, 128, 128, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up2 = UpResNet(128, 64, 64, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up3 = UpResNet(64, 64, 64, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up4 = OutResNet(64, n_classes, dropout_rate=dropout_rate)
        
    def forward(self, outputs): 
        y1 = self.up1(outputs[-1], outputs[-2])
        y2 = self.up2(y1, outputs[-3])
        y3 = self.up3(y2, outputs[-4])
        y4 = self.up4(y3)
        
        return y4
    
class MultiGPU_ResNetUNet_with_comm(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape, num_comm_fmaps, devices, depth=3, subdom_dist=(2, 2),
                 bilinear=False, comm=True, complexity=32, dropout_rate=0.1, kernel_size=5, padding=2, 
                 communicator_type=None, resnet_type = "resnet18", comm_network_but_no_communication=False):
        super(MultiGPU_ResNetUNet_with_comm, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.num_comm_fmaps = num_comm_fmaps
        self.devices = devices
        self.depth = depth
        self.subdom_dist = subdom_dist
        self.nx, self.ny = subdom_dist
        self.bilinear = bilinear
        self.comm = comm
        self.complexity = complexity
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.padding = padding
        self.communicator_type = communicator_type
        self.resnet_type = resnet_type
        self.comm_network_but_no_communication = comm_network_but_no_communication

        self.init_encoders(resnet_type=resnet_type)
        self.init_decoders()
        
        if self.comm:
            self.communication_network = CNNCommunicator(in_channels=num_comm_fmaps, out_channels=num_comm_fmaps,
                                                         dropout_rate=dropout_rate, kernel_size=kernel_size, padding=padding).to(devices[0])

    def init_encoders(self, resnet_type):
        encoder = ResNetEncoder(resnet_type=resnet_type, extra_conv=True, n_channels=self.n_channels, depth=self.depth, 
                                complexity=self.complexity, dropout_rate=self.dropout_rate)
        self.encoders = nn.ModuleList([copy.deepcopy(encoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])
        for encoder_module in self.encoders:
            # Fix parameters for the blocks
            for param in encoder_module.block1.parameters():
                param.requires_grad_(False)
            for param in encoder_module.block2.parameters():
                param.requires_grad_(False)
            for param in encoder_module.block3.parameters():
                param.requires_grad_(False)
            for param in encoder_module.block4.parameters():
                param.requires_grad_(False)
            # Keep parameters trainable for the extra_conv layer
            for param in encoder_module.conv.parameters():
                param.requires_grad_(True)
                

        
    def init_decoders(self):
        decoder = ResNetDecoder(n_channels=self.n_channels, n_classes=self.n_classes,
                          dropout_rate=self.dropout_rate)
        self.decoders = nn.ModuleList([copy.deepcopy(decoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def _select_device(self, index):
        return self.devices[index % len(self.devices)]
    
    def _synchronize_all_devices(self):
        for device in self.devices:
            torch.cuda.synchronize(device=device)
        
    def _get_list_index(self, i, j):
        return i * self.ny + j
    
    def _get_grid_index(self, index):
        return index // self.ny, index % self.ny
    
    def concatenate_tensors(self, tensors):
        concatenated_tensors = []
        for i in range(self.nx):
            column_tensors = []
            for j in range(self.ny):
                index = self._get_list_index(i, j)
                column_tensors.append(tensors[index].to(self._select_device(0)))
            concatenated_row = torch.cat(column_tensors, dim=2)
            concatenated_tensors.append(concatenated_row)

        return torch.cat(concatenated_tensors, dim=3)

    def _split_concatenated_tensor(self, concatenated_tensor):
        subdomain_tensors = []
        subdomain_height = concatenated_tensor.shape[3] // self.nx
        subdomain_width = concatenated_tensor.shape[2] // self.ny

        for i in range(self.nx):
            for j in range(self.ny):
                subdomain = concatenated_tensor[:, :, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors
        
    def forward(self, input_image_list):
        assert len(input_image_list) == self.nx * self.ny, "Number of input images must match the device grid size (nx x ny)."
        
        # Send to correct device and pass through encoder
        input_images_on_devices = [input_image.to(self._select_device(index)) for index, input_image in enumerate(input_image_list)]
        outputs_encoders = [self.encoders[index](input_image) for index, input_image in enumerate(input_images_on_devices)]
        
        inputs_decoders = [[x.clone() for x in y] for y in outputs_encoders]
        # Do the communication step. Replace the encoder outputs by the communication output feature maps
        if self.comm:
            if not self.comm_network_but_no_communication:
                communication_input = self.concatenate_tensors([output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders])
                communication_output = self.communication_network(communication_input)
                communication_output_split = self._split_concatenated_tensor(communication_output)
                
                for idx, output_communication in enumerate(communication_output_split):
                   inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication
            
            elif self.comm_network_but_no_communication:
                communication_inputs = [output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders]
                communication_outputs = [self.communication_network(comm_input.to(self.devices[0])) for comm_input in communication_inputs]
                
                for idx, output_communication in enumerate(communication_outputs):
                    inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication.to(self._select_device(idx))
                    
        # Do the decoding step
        outputs_decoders = [self.decoders[index](output_encoder) for index, output_encoder in enumerate(inputs_decoders)]
        prediction = self.concatenate_tensors(outputs_decoders)
               
        return prediction

    def save_weights(self, save_path):
        state_dict = {
            'encoder_state_dict': [self.encoders[0].state_dict()],
            'decoder_state_dict': [self.decoders[0].state_dict()]
        }
        if self.comm:
            state_dict['communication_network_state_dict'] = self.communication_network.state_dict()
        torch.save(state_dict, save_path)

    def load_weights(self, load_path, device="cuda:0"):
        checkpoint = torch.load(load_path, map_location=device)
        encoder_state = checkpoint['encoder_state_dict'][0]
        decoder_state = checkpoint['decoder_state_dict'][0]
        
        for encoder in self.encoders:
            encoder.load_state_dict(encoder_state)
        for decoder in self.decoders:
            decoder.load_state_dict(decoder_state)
        if self.comm and 'communication_network_state_dict' in checkpoint:
            self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
            
    # def save_weights(self, save_path):
    #     state_dict = {
    #         'encoder_state_dict': [encoder.state_dict() for encoder in self.encoders],
    #         'decoder_state_dict': [decoder.state_dict() for decoder in self.decoders]
    #     }
    #     if self.comm:
    #         state_dict['communication_network_state_dict'] = self.communication_network.state_dict()
    #     torch.save(state_dict, save_path)

    # def load_weights(self, load_path, device="cuda:0"):
    #     checkpoint = torch.load(load_path, map_location=device)
    #     for encoder, encoder_state in zip(self.encoders, checkpoint['encoder_state_dict']):
    #         encoder.load_state_dict(encoder_state)
    #     for decoder, decoder_state in zip(self.decoders, checkpoint['decoder_state_dict']):
    #         decoder.load_state_dict(decoder_state)
    #     if self.comm and 'communication_network_state_dict' in checkpoint:
    #         self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
