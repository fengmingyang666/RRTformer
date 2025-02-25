import torch
import numpy as np
import json
import argparse
from rrt_star_3d import RRTStar3D as RRT
from torchinfo import summary

def main_demo_plan_3d(env_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../checkpoint/3d/rg_3d.pt", map_location=device)
    summary(model, [(2,500,3),(2,50,50,50)])
    with open(env_file, 'r') as file:
        envs = json.load(file)
    env_idx = np.random.randint(0,len(envs))
    env = envs[env_idx]
    start = env['start'][0]
    goal = env['goal'][0]
    map_dim = env['env_dims']
    obstacle_list = []
    for circle in env['ball_obstacles']:
        obstacle_list.append((circle[0], circle[1], circle[2], circle[3]))
    rrt = RRT(start=start, goal=goal, map_dim=map_dim, obstacle_list=obstacle_list,search_radius=40)
    path,_,_,_,_ = rrt.plan_mixed(model=model, device=device)
    rrt.plot_path(path, title="Mixed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Demo Planning')
    parser.add_argument('--env_file', type=str, required=True, help='Path to the environment file')
    args = parser.parse_args()
    main_demo_plan_3d(args.env_file)
