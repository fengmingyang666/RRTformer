import torch
import torch.nn as nn
import torch.optim as optim
from model_3d import TransformerNodeSampler3D
from dataset import get_data_loader
from tqdm import tqdm
from torchinfo import summary
import sys
import numpy as np
import wandb
import os
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, data_loader, epochs=100, lr=0.001, device='cpu', checkpoint_dir='./checkpoint'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            pbar.set_description('Epoch {}/{}'.format(epoch, epochs))
            for batch_idx, (node_positions, target_points, env_maps) in tqdm(enumerate(data_loader), total=len(data_loader)):
                node_positions = node_positions.to(device) 
                target_points = target_points.to(device) 
                env_maps = env_maps.to(device) 
                optimizer.zero_grad()
                outputs = model(node_positions, env_map=env_maps)
                loss = mse_loss(outputs, target_points)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            pbar.set_postfix(loss='{:.3f}'.format(avg_loss))
            wandb.log({"loss": avg_loss})
            pbar.update(1)
            if (epoch % 1000 == 0):
                torch.save(model, os.path.join(checkpoint_dir, 'model_3d_{}.pt'.format(int(epoch / 1000))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D Model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/3d', help='Directory to save checkpoints')
    args = parser.parse_args()

    wandb.init(
        project="RRT_Net_3D",
        name="rrt_net_3d",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dataset": args.dataset_path,
        })

    data_loader = get_data_loader(args.dataset_path, batch_size=args.batch_size)
    model = TransformerNodeSampler3D(d_model=64, nhead=8, num_encoder_layers=6, depth=50, height=50, width=50)
    train_model(model, data_loader, epochs=args.epochs, lr=args.lr, device=device, checkpoint_dir=args.checkpoint_dir)
    wandb.finish()
    torch.save(model, os.path.join(args.checkpoint_dir, 'model_3d.pt'))
    print("model save done")
