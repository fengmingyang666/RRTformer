import json
import os
import cv2
import numpy as np
import argparse

def generate_env(img_height, img_width, circle_obstacles):
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    for circ in circle_obstacles:
        x, y, r = circ
        cv2.circle(env_img, (x, 100-y), r, (0, 0, 0), -1)
    binary_env = np.zeros((img_height, img_width), dtype=int)
    binary_env[env_img[:, :, 0] != 255] = 1
    return env_img, binary_env

def main(envs_file, output_dir):
    with open(envs_file, 'r') as f:
        envs_data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    for idx, env in enumerate(envs_data):
        img_height = env['env_dims'][0]
        img_width = env['env_dims'][1]
        circle_obstacles = env['circle_obstacles']
        env_img, binary_env = generate_env(img_height, img_width, circle_obstacles)
        output_path = os.path.join(output_dir, f'{idx}.png')
        cv2.imwrite(output_path, env_img)
        print(f'Environment {idx} saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Environment Images')
    parser.add_argument('--envs_file', type=str, required=True, help='Path to the environment file')
    parser.add_argument('--output_dir', type=str,default='../data/random_2d/test/env_fsg', help='Path to the output directory')
    args = parser.parse_args()
    main(args.envs_file, args.output_dir)


