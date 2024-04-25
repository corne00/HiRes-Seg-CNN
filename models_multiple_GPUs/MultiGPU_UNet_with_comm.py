import torch
import torch.nn as nn
import copy
from .sub_modules import CNNCommunicator, Encoder, Decoder

class MultiGPU_UNet_with_comm(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape, num_comm_fmaps, devices, depth=3, subdom_dist=(2, 2),
                 bilinear=False, comm=True, complexity=32, dropout_rate=0.1, kernel_size=5, padding=2, communicator_type=None):
        super(MultiGPU_UNet_with_comm, self).__init__()

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

        self.init_encoders()
        self.init_decoders()
        
        if self.comm:
            self.communication_network = CNNCommunicator(in_channels=num_comm_fmaps, out_channels=num_comm_fmaps,
                                                         dropout_rate=dropout_rate, kernel_size=kernel_size, padding=padding).to(devices[0])

    def init_encoders(self):
        encoder = Encoder(n_channels=self.n_channels, depth=self.depth, complexity=self.complexity,
                          dropout_rate=self.dropout_rate)
        self.encoders = nn.ModuleList([copy.deepcopy(encoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def init_decoders(self):
        decoder = Decoder(n_channels=self.n_channels, depth=self.depth, n_classes=self.n_classes,
                          complexity=self.complexity, dropout_rate=self.dropout_rate)
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

        # Do the communication step. Replace the encoder outputs by the communication output feature maps
        if self.comm:
            communication_input = self.concatenate_tensors([output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders])
            communication_output = self.communication_network(communication_input)
            communication_output_split = self._split_concatenated_tensor(communication_output)
            
            for idx, output_communication in enumerate(communication_output_split):
                outputs_encoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication

        # Do the decoding step
        outputs_decoders = [self.decoders[index](output_encoder) for index, output_encoder in enumerate(outputs_encoders)]
        prediction = self.concatenate_tensors(outputs_decoders)
               
        return prediction

    def save_weights(self, save_path):
        state_dict = {
            'encoder_state_dict': [encoder.state_dict() for encoder in self.encoders],
            'decoder_state_dict': [decoder.state_dict() for decoder in self.decoders]
        }
        if self.comm:
            state_dict['communication_network_state_dict'] = self.communication_network.state_dict()
        torch.save(state_dict, save_path)

    def load_weights(self, load_path):
        checkpoint = torch.load(load_path)
        for encoder, encoder_state in zip(self.encoders, checkpoint['encoder_state_dict']):
            encoder.load_state_dict(encoder_state)
        for decoder, decoder_state in zip(self.decoders, checkpoint['decoder_state_dict']):
            decoder.load_state_dict(decoder_state)
        if self.comm and 'communication_network_state_dict' in checkpoint:
            self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
