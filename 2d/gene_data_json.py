import yaml
import json
import random
import os
import argparse

def generate_random_circle(radius_range, height, width):
    radius = random.randint(*radius_range)
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    return [x,y,radius]

def generate_env(config):
    height = config['env_height']
    width = config['env_width']
    num_circles = random.randint(*config['num_circles_range'])
    goal = [random.randint(0, width), random.randint(0, height)]
    while goal[0] < 50 and goal[1] < 50:
        goal = [random.randint(0, width), random.randint(0, height)]
    env = {
        "env_dims":[width,height],
        "rectangle_obstacles": [],
        "circle_obstacles": [generate_random_circle(config['circle_radius_range'], height, width) for _ in range(num_circles)],
        "start": [[5, 5]],
        "goal": [[90,90]],
        "astar_time": [1]
    }
    return env

def main(yaml_path, json_path, mode):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    if mode=='train':
        envs = [generate_env(config) for _ in range(config['train_env_size'])]
    if mode=='test':
        envs = [generate_env(config) for _ in range(config['test_env_size'])]
    with open(json_path, 'w') as file:
        json.dump(envs, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Environment Data')
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    args = parser.parse_args()
    main(args.yaml_path, args.json_path, args.mode)