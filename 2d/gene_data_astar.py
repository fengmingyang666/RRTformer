import numpy as np
from tqdm import tqdm
import json
import time
import argparse
from astar import AStarDatasetGenerator
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_env(env):
    map_dim = env['env_dims']
    start = env['start'][0]
    goal = env['goal'][0]
    obstacle_list = []
    for circle in env['circle_obstacles']:
        obstacle_list.append([circle[0], circle[1], circle[2]])
    obstacle_list = np.array(obstacle_list)
    astar_gen = AStarDatasetGenerator(start=start, goal=goal, map_dim=map_dim, obstacle_list=obstacle_list)
    start_time = time.time()
    astar_path = astar_gen.plan()
    end_time = time.time()
    if astar_path is not None:
        env['astar_time'] = end_time - start_time
        return env, astar_gen.dataset, astar_gen.dataset_path
    return None

def generate_data_from_envs(envs_file, output_path_file):
    with open(envs_file, 'r') as file:
        envs = json.load(file)
    dataset = []
    dataset_path = []
    valid_envs = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_env, env): env for env in envs}
        for future in tqdm(as_completed(futures), total=len(envs)):
            result = future.result()
            if result is not None:
                env, data, path = result
                valid_envs.append(env)
                dataset.extend(data)
                dataset_path.extend(path)
    print("valid envs: ", len(valid_envs))
    with open(envs_file, 'w') as file:
        json.dump(valid_envs, file)
    dataset_path = np.array(dataset_path, dtype=object)
    np.save(output_path_file, dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate A* Data')
    parser.add_argument('--envs_file', type=str, required=True, help='Path to the environment file')
    parser.add_argument('--output_path_file', type=str, default= '../data/astar_2d.npy', help='Path to the output file')
    args = parser.parse_args()
    generate_data_from_envs(args.envs_file, args.output_path_file)