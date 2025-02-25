import torch
import numpy as np
import json
from rrt_star import RRTStar as RRT
import pickle
import argparse
from os.path import join, exists
import time
from copy import copy
def get_path_len(path):
    if path is None or len(path)==0:
        return np.inf
    else:
        path = np.array(path)
        path_disp = path[1:]-path[:-1]
        return np.linalg.norm(path_disp,axis=1).sum()

def main_evaluate(env_file):
    
    result_folderpath = 'evaluation'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../checkpoint/2d/rg_100.pt", map_location=device)
    with open(env_file, 'r') as file:
        envs = json.load(file)
    num_problem = len(envs)
    result_filepath = join(result_folderpath, 'random_2d-ours-our_model-{}.pickle'.format(num_problem))
    if not exists(result_filepath):
        env_result_config_list = []
    else:
        with open(result_filepath, 'rb') as f:
            env_result_config_list = pickle.load(f)
    eval_start_time = time.time()
    print(env_result_config_list)
    for idx,env in enumerate(envs):
        if idx < len(env_result_config_list):
            time_left = (time.time() - eval_start_time) * (num_problem / (idx + 1) - 1) / 60
            print("Evaluated {0}/{1} in the loaded file, remaining time: {2} min for {3}".format(idx + 1, num_problem, int(time_left), 'random_2d-ours-our_model-num_problem'))
            continue
        start = env['start'][0]
        goal = env['goal'][0]
        map_dim = env['env_dims']
        obstacle_list = []
        for circle in env['circle_obstacles']:
            obstacle_list.append((circle[0], circle[1], circle[2]))
        rrt = RRT(start=start, goal=goal, map_dim=map_dim, obstacle_list=obstacle_list, max_iter=1000,search_radius=20)
        path_len = np.inf
        plot_time = 0
        cnt=0
        tt1 = time.time()
        while path_len==np.inf and cnt<100:
            path,num_node,times,iters,refine_cost = rrt.plan_mixed(model=model, device=device)
            path_len = get_path_len(path)
            cnt += 1
        tt2 = time.time()
        
        env_result_config = copy(env)
        env_result_config['result'] = [path_len]
        env_result_config['num nodes'] = num_node
        env_result_config['time'] = times
        env_result_config['total time'] = tt2-tt1-plot_time
        env_result_config['iters'] = iters
        env_result_config['refine cost'] = refine_cost
        env_result_config_list.append(env_result_config)
        

        with open(result_filepath, 'wb') as f:
            pickle.dump(env_result_config_list, f)
        time_left = (time.time() - eval_start_time) * (num_problem / (idx + 1) - 1) / 60
        print("Metric: length-{}, nodes-{}, time-{}".format(env_result_config['result'],env_result_config['num nodes'],env_result_config['time']))
        print("Evaluated {0}/{1}, remaining time: {2} min for {3}".format(idx + 1, num_problem, int(time_left), 'random_2d-ours-our_model-num_problem'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Planning')
    parser.add_argument('--env_file', type=str, required=True, help='Path to the environment file')
    args = parser.parse_args()
    main_evaluate(args.env_file)

