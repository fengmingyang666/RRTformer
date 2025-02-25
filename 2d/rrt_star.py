import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import time

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def get_path_len(path):
    if path is None or len(path)==0:
        return np.inf
    else:
        path = np.array(path)
        path_disp = path[1:]-path[:-1]
        return np.linalg.norm(path_disp,axis=1).sum()

class RRTStar:
    def __init__(self, start, goal, obstacle_list, map_dim, step_scale=0.03, goal_sample_rate=0.1, max_iter=1000,search_radius=10):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_dim = map_dim
        self.obstacle_list = obstacle_list
        self.step_size = int(np.sqrt(map_dim[0]**2+map_dim[1]**2)*step_scale)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.env_map = self.create_env_map()
        self.search_radius = search_radius


    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(0, self.map_dim[0]), random.uniform(0, self.map_dim[1]))
        else:
            return Node(self.goal.x, self.goal.y)

    def transformer_sample(self, model, device):
        if random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y)
        tree_nodes = list()
        for node in self.node_list:
            tree_nodes.append([node.x, node.y])
        tree_nodes_tensor = torch.tensor(tree_nodes, dtype=torch.float32).unsqueeze(0).to(device) 
        env_map_tensor = torch.tensor(self.env_map, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            next_sample_point = model(tree_nodes_tensor, env_map_tensor)
        node_next = next_sample_point.cpu().numpy()[0]
        return Node(node_next[0], node_next[1])

    def create_env_map(self):
        env_map = np.zeros(self.map_dim, dtype=np.uint8)
        for (ox, oy, size) in self.obstacle_list:
            for i in range(int(ox - size), int(ox + size)):
                for j in range(int(oy - size), int(oy + size)):
                    if 0 <= i < self.map_dim[0] and 0 <= j < self.map_dim[1]:
                        if np.linalg.norm((ox - i, oy - j)) <= size:
                            env_map[i, j] = 1

        return env_map

    def get_nearest_node_index(self, random_node):
        distances = [np.linalg.norm((node.x - random_node.x, node.y - random_node.y)) for node in self.node_list]
        return distances.index(min(distances))

    def steer(self, from_node, to_node):
        angle = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(angle)
        new_y = from_node.y + self.step_size * np.sin(angle)
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm((new_node.x - from_node.x, new_node.y - from_node.y))
        return new_node

    def is_collision(self, node):
        for (ox, oy, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            if dx**2 + dy**2 <= size**2:
                return True
        return False
    def get_nearby_nodes(self, new_node):
        nearby_nodes = []
        for node in self.node_list:
            if np.linalg.norm((node.x - new_node.x, node.y - new_node.y)) <= self.search_radius:
                nearby_nodes.append(node)
        return nearby_nodes

    def choose_parent(self, new_node, nearby_nodes):
        best_node = None
        min_cost = float('inf')
        for node in nearby_nodes:
            temp_node = self.steer(node, new_node)
            if not self.is_collision(temp_node) and temp_node.cost < min_cost:
                best_node = node
                min_cost = temp_node.cost
        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            temp_node = self.steer(new_node, node)
            if not self.is_collision(temp_node) and new_node.cost + np.linalg.norm((new_node.x - node.x, new_node.y - node.y)) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + np.linalg.norm((new_node.x - node.x, new_node.y - node.y))

    def generate_path(self, last_node):
        path = [[self.goal.x, self.goal.y]]
        node = last_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def find_shortest_path_in_tree(self):
        min_cost = float('inf')
        best_goal_node = None
        for node in self.node_list:
            if np.linalg.norm((node.x - self.goal.x, node.y - self.goal.y)) <= self.step_size:
                if node.cost < min_cost:
                    min_cost = node.cost
                    best_goal_node = node
        if best_goal_node:
            return self.generate_path(best_goal_node)
        else:
            return None

    def plan_mixed(self,model,device,alpha=0.5):
        path_len_list = []
        refine_path_len_list = []
        t1 = time.time()
        initial_path_found = False
        initial_path = None
        iter = 0

        for k in range(self.max_iter):
            if np.random.rand()>alpha:
                random_node = self.transformer_sample(model, device)
            else:
                random_node = self.get_random_node()
            nearest_node_index = self.get_nearest_node_index(random_node)
            nearest_node = self.node_list[nearest_node_index]
            new_node = self.steer(nearest_node, random_node)

            if not self.is_collision(new_node):
                nearby_nodes = self.get_nearby_nodes(new_node)
                new_node = self.choose_parent(new_node, nearby_nodes)
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, nearby_nodes)

                if np.linalg.norm((new_node.x - self.goal.x, new_node.y - self.goal.y)) <= self.step_size:
                    initial_path_found = True
                    initial_path = self.generate_path(new_node)
                    current_path_len = get_path_len(initial_path)
                    path_len_list.append(current_path_len)
                    print(f"Initial path found at iteration {k}")
                    iter = k
                    break

            current_path = self.generate_path(new_node) if initial_path_found else []
            current_path_len = get_path_len(current_path)
            path_len_list.append(current_path_len)
        num_node = len(self.node_list)
        t2 = time.time()
        refine_path_len_list.append((0, path_len_list[-1]))
        if not initial_path_found:
            return None, num_node, t2 - t1,iter, refine_path_len_list
        initial_path_len = path_len_list[-1]
        cost_min = initial_path_len
        start_time = time.time()
        last_record_time = start_time
        while time.time() - start_time < 20:
            if np.random.rand()>alpha:
                random_node = self.transformer_sample(model, device)
            else:
                random_node = self.get_random_node()
            nearest_node_index = self.get_nearest_node_index(random_node)
            nearest_node = self.node_list[nearest_node_index]
            new_node = self.steer(nearest_node, random_node)

            if not self.is_collision(new_node):
                nearby_nodes = self.get_nearby_nodes(new_node)
                new_node = self.choose_parent(new_node, nearby_nodes)
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, nearby_nodes)
            current_path = self.find_shortest_path_in_tree()
            current_path_len = get_path_len(current_path)
            if current_path and current_path_len<cost_min:
                cost_min = current_path_len
            
                path_len_list.append(current_path_len)
            current_time = time.time()
            if current_time - last_record_time >= 2:
                running_time = current_time - start_time
                last_record_time = current_time
                refine_path_len_list.append((running_time, cost_min))
                print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}".format(
                    int(running_time), 20, cost_min, initial_path_len))

        return initial_path, num_node, t2 - t1,iter, refine_path_len_list


    def plot_path(self,path, title="RRTStar Path with Obstacles"):
        plt.figure(figsize=(8, 8))

        for (ox, oy, radius) in self.obstacle_list:
            circle = plt.Circle((ox, oy), radius, color='red')
            plt.gca().add_patch(circle)
        tree_nodes = self.node_list
        tree_nodes_x = list()
        tree_nodes_y = list()
        for node in tree_nodes:
            tree_nodes_x.append(node.x)
            tree_nodes_y.append(node.y)
        plt.scatter(tree_nodes_x, tree_nodes_y, c='blue', label='RRT Nodes')
        if path != None:
            tree_nodes = path
            tree_nodes_x = list()
            tree_nodes_y = list()
            for node in tree_nodes:
                tree_nodes_x.append(node[0])
                tree_nodes_y.append(node[1])
            plt.plot(tree_nodes_x, tree_nodes_y, c='green', linewidth=2, label='Optimized Path')
        plt.scatter(self.start.x, self.start.y, c='green', label='Start')
        plt.scatter(self.goal.x, self.goal.y, c='purple',label='Goal')
        plt.xlim(0, self.map_dim[0])
        plt.ylim(0, self.map_dim[1])
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()
        plt.close()
