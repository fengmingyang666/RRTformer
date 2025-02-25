import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.register_buffer('position_encoding', self.generate_positional_encoding(height, width))

    def generate_positional_encoding(self, height, width):
        position_encoding = np.zeros((height, width, self.d_model))

        for y in range(height):
            for x in range(width):
                for i in range(0, self.d_model, 2):
                    position_encoding[y, x, i] = np.sin(x / (10000 ** (i / self.d_model)))
                    position_encoding[y, x, i + 1] = np.cos(y / (10000 ** ((i + 1) / self.d_model)))

        return torch.tensor(position_encoding, dtype=torch.float32).permute(2, 0, 1)  # (d_model, height, width)

    def forward(self, x):
        batch_size, d_model, height, width = x.size()
        if self.position_encoding.size(1) != height or self.position_encoding.size(2) != width:
            position_encoding_resized = F.interpolate(self.position_encoding.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
        else:
            position_encoding_resized = self.position_encoding
        return x + position_encoding_resized.to(x.device)
    def visualize_positional_encoding(self):
        position_encoding_np = self.position_encoding.cpu().numpy()
        for i in range(self.d_model):
            plt.imshow(position_encoding_np[i, :, :], cmap='viridis')
            plt.colorbar()
            plt.title(f'Positional Encoding - Channel {i}')
            plt.savefig(f'positional_encoding_channel_{i}.png')
            plt.close()


class TransformerNodeSampler(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, height, width, dim_feedforward=512):
        super(TransformerNodeSampler, self).__init__()
        self.d_model = d_model

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1)
        )

        self.pos_encoder = PositionalEncoding(d_model, height, width)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.input_embedding = nn.Linear(2, d_model)  
        self.output_layer = nn.Linear(d_model, 2)     

    def forward(self, node_positions, env_map):

        batch_size = node_positions.size(0)
        num_nodes = node_positions.size(1)
        env_map = env_map.unsqueeze(1)  
        env_features = self.conv(env_map)  
        # self.save_feature_maps(env_features, 'env_features') # save feature to an image
        pos_encoded_map = self.pos_encoder(env_features) 
        embedded_nodes = self.input_embedding(node_positions)
        embedded_nodes = embedded_nodes.permute(1, 0, 2)
        node_x = node_positions[:, :, 0].unsqueeze(2)
        node_y = node_positions[:, :, 1].unsqueeze(2)
        node_x_norm = node_x / (env_map.size(3) - 1)
        node_y_norm = node_y / (env_map.size(2) - 1)
        grid = torch.cat([node_x_norm, node_y_norm], dim=-1) 
        grid = grid.unsqueeze(2) 
        sampled_env_features = F.grid_sample(pos_encoded_map, grid, align_corners=True)
        sampled_env_features = sampled_env_features.squeeze(-1).permute(2,0,1)
        transformer_input = embedded_nodes + sampled_env_features 
        transformer_output = self.transformer_encoder(transformer_input) 
        next_sample_point = self.output_layer(transformer_output[-1]) 

        return next_sample_point
    
    def save_feature_maps(self, feature_maps, name):
        feature_maps_np = feature_maps.detach().cpu().numpy()
        for i in range(feature_maps_np.shape[1]):
            plt.imshow(np.flipud(feature_maps_np[0, i, :, :]), cmap='viridis')
            plt.colorbar()
            plt.title(f'{name}')
            plt.savefig(f'{name}_feature_map_{i}.png')
            plt.close()
