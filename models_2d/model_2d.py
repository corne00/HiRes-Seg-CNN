import torch.nn
from .model2d_components import DoubleConv, Up, Down, CNNCommunicator, OutConv
import torch.nn as nn
import torch.nn.functional as F

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
        #self.down3 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))

        factor = 2 if bilinear else 1
        
        self.up1 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.up2 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        #self.up3 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.outc = (OutConv(in_channels, out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        #x = self.up3(x, x1)

        output = self.outc(x)

        return output

class UNet_2D_conv_comm_varying_depth(nn.Module):
    def __init__(self, n_channels, n_classes, inp_size, inp_comm, outp_comm, device, sizex=None, sizey=None,
                 depth = 3, bilinear=False, comm=True, n_complexity = 32, dropout_rate = 0.1, kernel_size=3, padding=1, communicator_type=None):
        super(UNet_2D_conv_comm_varying_depth, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inp_size = inp_size

        self.inp_comm = inp_comm
        self.outp_comm = outp_comm

        self.sizex = sizex
        self.sizey = sizey
        self.communicator_type = communicator_type

        if comm:
            self.name = f"2D_UNet_2D_comm_depth_{depth}_subdom_{inp_size}_inp_{inp_comm}_outp_{outp_comm}"
        else:
            self.name = f"2D_UNet_2D_no_comm_depth_{depth}_subdom_{inp_size}_inp_{inp_comm}_outp_{outp_comm}"

        self.comm = comm

        ### SET UP THE UNET
        self.inc = (DoubleConv(n_channels, n_complexity, dropout_rate=dropout_rate))  # Reduced from 64 to 16
        self.contraction = nn.ModuleList()
        if self.comm: 
            if self.communicator_type == "UNet":
                self.communication_network = UNetCommunicator(inp_comm, outp_comm, dropout_rate=dropout_rate, bilinear=False)
            else:
                self.communication_network = CNNCommunicator(inp_comm, outp_comm, dropout_rate=dropout_rate, kernel_size=kernel_size, padding=padding)

        self.expansion = nn.ModuleList()

        factor = 2 if bilinear else 1
        channels = [n_complexity*2**i for i in range(depth+1)]

        for i in range(depth):
            self.contraction.append(Down(channels[i], channels[i+1], dropout_rate=dropout_rate))
            self.expansion.append(Up(channels[-1-i] // factor, channels[-2-i] // factor, bilinear=bilinear, dropout_rate=dropout_rate))

        self.outc = (OutConv(n_complexity, n_classes))  

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
                x1 = self.inc(data_tensors[j][i])
                contracting_tensors[j][i].append(x1)

                for contracting_layer in self.contraction:
                    x1 = contracting_layer(x1)
                    contracting_tensors[j][i].append(x1)

        # Do the communication step if wanted
        if self.comm:
            input_communication = [[] for _ in range(n_y)]

            for j in range(n_y):
                for i in range(n_x):
                    last_feature_maps = contracting_tensors[j][i][-1][:,-self.inp_comm:,:,:]
                    input_communication[j].append(last_feature_maps)
                    #print(last_feature_maps.shape)

            input_communication_inter = []
            for j in range(n_y):
                input_communication_inter.append(torch.cat(input_communication[j], dim=3))

            input_communication = torch.cat(input_communication_inter, dim=2)
            output_communication = self.communication_network(input_communication)

            for j in range(n_y):
                for i in range(n_x):
                    contracting_tensors[j][i][-1] = torch.cat([contracting_tensors[j][i][-1][:, :-self.outp_comm, :, :],
                                                        output_communication[:,:,(j*self.sizey):(j+1)*self.sizey,(i*self.sizex):(i+1)*self.sizex]], dim=1)

        # Generate a list containing the outputs
        output_logits = [[] for _ in range(n_y)]

        for j in range(n_y):
            for i in range(n_x):
                x2 = contracting_tensors[j][i][-1]

                for k, expansion_layer in enumerate(self.expansion):
                    x2 = expansion_layer(x2, contracting_tensors[j][i][-2-k])
                logits = self.outc(x2)
                output_logits[j].append(logits)

        output_logits_inter = []
        for j in range(n_y):
            output_logits_inter.append(torch.cat(output_logits[j], dim=3))

        output_logits = torch.cat(output_logits_inter, dim=2)

        return output_logits