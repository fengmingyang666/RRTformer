import torch
import numpy as np
import json
import argparse
from rrt_star import RRTStar as RRT
from torchinfo import summary

def main_demo_plan(env_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../checkpoint/2d/rg_100.pt", map_location=device)
    summary(model, [(2,500,2),(2,100,100)])
    with open(env_file, 'r') as file:
        envs = json.load(file)
    env = envs[0]
    start = env['start'][0]
    goal = env['goal'][0]
    map_dim = env['env_dims']
    obstacle_list = []
    for circle in env['circle_obstacles']:
        obstacle_list.append((circle[0], circle[1], circle[2]))

    rrt = RRT(start=start, goal=goal, map_dim=map_dim, obstacle_list=obstacle_list,search_radius=20)
    path,_,_,_ = rrt.plan_mixed(model=model, device=device)
    rrt.plot_path(path, title="Mixed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo Planning')
    parser.add_argument('--env_file', type=str, required=True, help='Path to the environment file')
    args = parser.parse_args()
    main_demo_plan(args.env_file)
