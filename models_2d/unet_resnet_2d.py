import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from .model2d_components import DoubleConv, Up, Down, CNNCommunicator, UNetCommunicator, OutConv, UpResNet



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


class ResNetUNet_2D(nn.Module):
    def __init__(self, n_channels, n_classes, inp_size, inp_comm, outp_comm,
                 resnet_type="resnet18", bilinear=False, comm=True, dropout_rate = 0.1, 
                 communicator_type=None, extra_conv_bottleneck=True):

        super(ResNetUNet_2D, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inp_size = inp_size

        self.inp_comm = inp_comm
        self.outp_comm = outp_comm
        
        self.communicator_type = communicator_type
        self.dropout_rate = dropout_rate
        self.communicator_type = communicator_type

        self.extra_conv_bottleneck = extra_conv_bottleneck

        self.comm = comm

        if comm:
            self.name = f"ResNetUNet_2D_comm_{communicator_type}_{inp_size}_{inp_comm}_{outp_comm}_{resnet_type}"
            
            if communicator_type == None:
                self.communication_network = CNNCommunicator(inp_comm, outp_comm, dropout_rate=dropout_rate, kernel_size=5, padding=3)      
            elif communicator_type == "unet":
                self.communication_network = UNetCommunicator(inp_comm, outp_comm, dropout_rate=dropout_rate, bilinear=False)
            else:
                print("Selected wrong communicator type! Set to default CNNCommunicator")
                self.communication_network = CNNCommunicator(inp_comm, outp_comm, dropout_rate=dropout_rate, kernel_size=5, padding=3)      
                        
        
        if not comm:
            self.name = f"ResNetUNet_2D_no_comm_{communicator_type}_{inp_size}_{inp_comm}_{outp_comm}_{resnet_type}"


        self.resnet_type = resnet_type

        if self.resnet_type == "resnet50":
            base_model = torchvision.models.resnet50(weights="DEFAULT")
            self.channel_distribution = [3, 64, 256, 512, 1024]
        elif self.resnet_type == "resnet34":
            base_model = torchvision.models.resnet34(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        elif self.resnet_type == "resnet18":
            base_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        else:
            print("Pretrained network not found. Using pretrained ResNet 18.")
            base_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]

        self.encoder_layers = list(base_model.children())
 
        
        self.block1 = nn.Sequential(*self.encoder_layers[:3])  # 3   /  3  -->  64 / 64
        self.block2 = nn.Sequential(*self.encoder_layers[3:5]) # 64  / 64  -->  64 / 256
        self.block3 = nn.Sequential(*self.encoder_layers[5])   # 64  / 256 --> 128 / 512 
        self.block4 = nn.Sequential(*self.encoder_layers[6])   # 128 / 512 --> 256 / 1024
        
        # Fix the parameters in the pretrained network
        blocks_to_fix = [self.block1, self.block2, self.block3, self.block4]

        for block in blocks_to_fix:
            for param in block.parameters():
                param.requires_grad_(False)

        if extra_conv_bottleneck:
            self.conv1 = DoubleConv(self.channel_distribution[-1], self.channel_distribution[-1], dropout_rate=dropout_rate)
            #self.conv2 = DoubleConv(self.channel_distribution[-1], self.channel_distribution[-1], dropout_rate=dropout_rate)

        self.up1 = UpResNet(self.channel_distribution[-1], 
                            self.channel_distribution[-2], 
                            self.channel_distribution[-2],
                            bilinear=True, dropout_rate=dropout_rate)
        self.up2 = UpResNet(self.channel_distribution[-2], 
                            self.channel_distribution[-3], 
                            self.channel_distribution[-3],
                            bilinear=True, dropout_rate=dropout_rate)
        self.up3 = UpResNet(self.channel_distribution[-3], 
                            self.channel_distribution[-4], 
                            self.channel_distribution[-4],
                            bilinear=True, dropout_rate=dropout_rate)
        self.up4 = OutResNet(self.channel_distribution[-4], 
                              n_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        # Compute the subdomain width, height and the corresponding number of subdomains
        subdomain_width = self.inp_size[0]
        subdomain_height = self.inp_size[1]
        n_x = x.shape[3] // subdomain_width
        n_y = x.shape[2] // subdomain_height

        # Create a list of data tensors contains one subdomain
        data_tensors = [[] for _ in range(n_y)]
        for i in range(n_x):
            for j in range(n_y):
                data_tensors[j].append(x[:,:,j*subdomain_height:(j+1)*subdomain_height, i* subdomain_width:(i+1)*subdomain_width])

        # Create a list storing the outputs of the contraction path (necessary for skip connections)
        contracting_tensors = [[[] for _ in range(n_x)] for _ in range(n_y)]

        for j in range(n_y):
            for i in range(n_x):
                x1 = self.block1(data_tensors[j][i])
                x2 = self.block2(x1)
                x3 = self.block3(x2)
                if self.extra_conv_bottleneck:
                    x4 = self.conv1(self.block4(x3))
                else:
                    x4 = self.block4(x3)

                # print("x : ", x.shape)
                # print("x1: ", x1.shape)
                # print("x2: ", x2.shape)
                # print("x3: ", x3.shape)
                # print("x4: ", x4.shape)

                contracting_tensors[j][i].append(x1)
                contracting_tensors[j][i].append(x2)
                contracting_tensors[j][i].append(x3)
                contracting_tensors[j][i].append(x4)

        # Do the communication step if wanted
        if self.comm:
            input_communication = [[] for _ in range(n_y)]

            for j in range(n_y):
                for i in range(n_x):
                    last_feature_maps = contracting_tensors[j][i][-1][:,-self.inp_comm:,:,:]
                    input_communication[j].append(last_feature_maps)
                    #print(last_feature_maps.shape)

            sizex, sizey = input_communication[0][0].shape[3], input_communication[0][0].shape[2]
            # print("sizex:", sizex)
            # print("sizey:", sizey)

            input_communication_inter = []
            for j in range(n_y):
                input_communication_inter.append(torch.cat(input_communication[j], dim=3))

            input_communication = torch.cat(input_communication_inter, dim=2)
            output_communication = self.communication_network(input_communication)

            for j in range(n_y):
                for i in range(n_x):
                    contracting_tensors[j][i][-1] = torch.cat([contracting_tensors[j][i][-1][:, :-self.outp_comm, :, :],
                                                        output_communication[:,:,(j*sizey):(j+1)*sizey,(i*sizex):(i+1)*sizex]], dim=1)

       # Generate a list containing the outputs
        output_logits = [[] for _ in range(n_y)]

        for j in range(n_y):
            for i in range(n_x):
                #print(contracting_tensors[j][i][-1].shape)
                #print(contracting_tensors[j][i][-2].shape)

                y1 = self.up1(contracting_tensors[j][i][-1], contracting_tensors[j][i][-2])
                y2 = self.up2(y1, contracting_tensors[j][i][-3])
                y3 = self.up3(y2, contracting_tensors[j][i][-4])
                y4 = self.up4(y3)

                # print("y1: ", y1.shape)
                # print("y2: ", y2.shape)
                # print("y3: ", y3.shape)
                # print("y4: ", y4.shape)

                logits = y4
                output_logits[j].append(logits)                

        output_logits_inter = []
        for j in range(n_y):
            output_logits_inter.append(torch.cat(output_logits[j], dim=3))

        output_logits = torch.cat(output_logits_inter, dim=2)

        return output_logits