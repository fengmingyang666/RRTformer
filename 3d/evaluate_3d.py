import torch
import numpy as np
import json
import argparse
from rrt_star_3d import RRTStar3D as RRTStar
import pickle
from os.path import join, exists
import time
from copy import copy

def get_path_len(path):
    if path is None or len(path) == 0:
        return np.inf
    else:
        path = np.array(path)
        path_disp = path[1:] - path[:-1]
        return np.linalg.norm(path_disp, axis=1).sum()

def main_evaluate_3d(env_file):
    result_folderpath = 'result/evaluation'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../checkpoint/3d/rg_3d.pt", map_location=device)
    with open(env_file, 'r') as file:
        envs = json.load(file)
    num_problem = len(envs)
    result_filepath = join(result_folderpath, 'random_3d-ours-our_model-{}.pickle'.format(num_problem))
    if not exists(result_filepath):
        env_result_config_list = []
    else:
        with open(result_filepath, 'rb') as f:
            env_result_config_list = pickle.load(f)
    eval_start_time = time.time()
    for idx, env in enumerate(envs):
        if idx < len(env_result_config_list):
            time_left = (time.time() - eval_start_time) * (num_problem / (idx + 1) - 1) / 60
            print("Evaluated {0}/{1} in the loaded file, remaining time: {2} min for {3}".format(idx + 1, num_problem, int(time_left), 'random_3d-ours-our_model-num_problem'))
            continue
        start = env['start'][0]
        goal = env['goal'][0]
        map_dim = env['env_dims']
        obstacle_list = []
        for sphere in env['ball_obstacles']:
            obstacle_list.append((sphere[0], sphere[1], sphere[2], sphere[3]))
        rrt_star = RRTStar(start=start, goal=goal, map_dim=map_dim, obstacle_list=obstacle_list, max_iter=1000,search_radius=40)
        path_len = np.inf
        cnt = 0
        tt1 = time.time()
        while path_len == np.inf and cnt < 100:
            path, num_node, times,iters, refine_cost = rrt_star.plan_mixed(model=model, device=device)
            path_len = get_path_len(path)
            cnt += 1
        tt2 = time.time()
    
        env_result_config = copy(env)
        env_result_config['result'] = [path_len]
        env_result_config['num nodes'] = num_node
        env_result_config['time'] = times
        env_result_config['total time'] = tt2 - tt1
        env_result_config['iters'] = iters
        env_result_config['refine cost'] = refine_cost
        env_result_config_list.append(env_result_config)
        
        with open(result_filepath, 'wb') as f:
            pickle.dump(env_result_config_list, f)
        time_left = (time.time() - eval_start_time) * (num_problem / (idx + 1) - 1) / 60
        print("Metric: length-{}, nodes-{}, time-{}".format(env_result_config['result'], env_result_config['num nodes'], env_result_config['time']))
        print("Evaluated {0}/{1}, remaining time: {2} min for {3}".format(idx + 1, num_problem, int(time_left), 'random_3d-ours-our_model-num_problem'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 3D Planning')
    parser.add_argument('--env_file', type=str, required=True, help='Path to the environment file')
    args = parser.parse_args()
    main_evaluate_3d(args.env_file)
