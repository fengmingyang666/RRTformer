import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, depth, height, width):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.register_buffer('position_encoding', self.generate_positional_encoding(depth, height, width))
    def generate_positional_encoding(self, depth, height, width):
        position_encoding = np.zeros((depth, height, width, self.d_model))
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    for i in range(0, self.d_model, 2):
                        position_encoding[z, y, x, i] = np.sin(x / (10000 ** (i / self.d_model)))
                        position_encoding[z, y, x, i + 1] = np.cos(y / (10000 ** ((i + 1) / self.d_model)))
        return torch.tensor(position_encoding, dtype=torch.float32).permute(3, 0, 1, 2) 
    def forward(self, x):
        batch_size, d_model, depth, height, width = x.size()
        if self.position_encoding.size(1) != depth or self.position_encoding.size(2) != height or self.position_encoding.size(3) != width:
            position_encoding_resized = F.interpolate(self.position_encoding.unsqueeze(0), size=(depth, height, width), mode='trilinear', align_corners=False).squeeze(0)
        else:
            position_encoding_resized = self.position_encoding
        return x + position_encoding_resized.to(x.device)

class TransformerNodeSampler3D(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, depth, height, width, dim_feedforward=512):
        super(TransformerNodeSampler3D, self).__init__()
        self.d_model = d_model

        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, d_model, kernel_size=3, padding=1)
        )

        self.pos_encoder = PositionalEncoding3D(d_model, depth, height, width)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.input_embedding = nn.Linear(3, d_model) 
        self.output_layer = nn.Linear(d_model, 3) 

    def forward(self, node_positions, env_map):
        batch_size = node_positions.size(0)
        num_nodes = node_positions.size(1)
        env_map = env_map.unsqueeze(1) 
        env_features = self.conv(env_map) 
        pos_encoded_map = self.pos_encoder(env_features)
        embedded_nodes = self.input_embedding(node_positions)
        embedded_nodes = embedded_nodes.permute(1, 0, 2)
        node_x = node_positions[:, :, 0].unsqueeze(2) 
        node_y = node_positions[:, :, 1].unsqueeze(2) 
        node_z = node_positions[:, :, 2].unsqueeze(2) 
        node_x_norm = node_x / (env_map.size(4) - 1) 
        node_y_norm = node_y / (env_map.size(3) - 1) 
        node_z_norm = node_z / (env_map.size(2) - 1) 
        grid = torch.cat([node_x_norm, node_y_norm, node_z_norm], dim=-1)
        grid = grid.unsqueeze(1)
        grid = grid.unsqueeze(1)
        sampled_env_features = F.grid_sample(pos_encoded_map, grid, align_corners=True) 
        sampled_env_features = sampled_env_features.squeeze(2).squeeze(2).permute(2, 0, 1) 
        transformer_input = embedded_nodes + sampled_env_features  
        transformer_output = self.transformer_encoder(transformer_input) 
        next_sample_point = self.output_layer(transformer_output[-1])
        return next_sample_point
    
    def save_feature_maps(self, feature_maps, name):
        feature_maps_np = feature_maps.detach().cpu().numpy()
        self.save_3d_feature_map(feature_maps_np, name)
    def save_3d_feature_map(self, feature_maps_np, name):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        depth, height, width = feature_maps_np.shape[2], feature_maps_np.shape[3], feature_maps_np.shape[4]
        x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
        feature_maps_np = (feature_maps_np - feature_maps_np.min()) / (feature_maps_np.max() - feature_maps_np.min())
        for channel in range(feature_maps_np.shape[1]):
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=feature_maps_np[0, channel, :, :, :].flatten(), cmap='viridis', marker='o', alpha=0.01)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f'{name} - 3D Feature Map')
            plt.savefig(f'{name}_{channel}_3d_feature_map.png')
        plt.close()
