import yaml
import json
import random
import os
import argparse

def generate_random_rectangle(width_range, height, width, depth):
    rect_width = random.randint(*width_range)
    rect_height = random.randint(*width_range)
    rect_depth = random.randint(*width_range)
    x = random.randint(0, width - rect_width)
    y = random.randint(0, height - rect_height)
    z = random.randint(0, depth - rect_depth)
    return {'type': 'rectangle', 'x': x, 'y': y, 'z': z, 'width': rect_width, 'height': rect_height, 'depth': rect_depth}

def generate_random_sphere(radius_range, height, width, depth):
    radius = random.randint(*radius_range)
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    z = random.randint(radius, depth - radius)
    return [x, y, z, radius]

def generate_env(config):
    height = config['xyz_max'][0]
    width = config['xyz_max'][1]
    depth = config['xyz_max'][2]
    num_spheres = random.randint(*config['num_balls_range'])
    goal = [random.randint(0, width), random.randint(0, height), random.randint(0, depth)]
    while goal[0] < 10 and goal[1] < 10 and goal[2] < 10:
        goal = [random.randint(0, width), random.randint(0, height), random.randint(0, depth)]
    env = {
        "env_dims": [width, height, depth],
        "box_obstacles": [],
        "ball_obstacles": [generate_random_sphere(config['ball_radius_range'], height, width, depth) for _ in range(num_spheres)],
        "start": [[5, 5, 5]],
        "goal": goal,
        "astar_time": [1]
    }
    return env

def main(yaml_path, json_path,mode):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    if mode=='train':
        envs = [generate_env(config) for _ in range(config['train_env_size'])]
    if mode=='test':
        envs = [generate_env(config) for _ in range(config['test_env_size'])]
    with open(json_path, 'w') as file:
        json.dump(envs, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D Environment Data')
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    args = parser.parse_args()
    main(args.yaml_path, args.json_path,args.mode)