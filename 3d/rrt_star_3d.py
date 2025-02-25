import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from mpl_toolkits.mplot3d import Axes3D
import time

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

def get_path_len(path):
    if path is None or len(path) == 0:
        return np.inf
    else:
        path = np.array(path)
        path_disp = path[1:] - path[:-1]
        return np.linalg.norm(path_disp, axis=1).sum()

class RRTStar3D:
    def __init__(self, start, goal, obstacle_list, map_dim, step_scale=0.116, goal_sample_rate=0.1, max_iter=1000, search_radius=40):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.map_dim = map_dim
        self.obstacle_list = obstacle_list
        self.step_size = int(np.sqrt(map_dim[0]**2 + map_dim[1]**2 + map_dim[2]**2) * step_scale)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.env_map = self.create_env_map()
        self.search_radius = search_radius

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(0, self.map_dim[0]), random.uniform(0, self.map_dim[1]), random.uniform(0, self.map_dim[2]))
        else:
            return Node(self.goal.x, self.goal.y, self.goal.z)

    def transformer_sample(self, model, device):
        if random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y, self.goal.z)
        tree_nodes = list()
        for node in self.node_list:
            tree_nodes.append([node.x, node.y, node.z])
        tree_nodes_tensor = torch.tensor(tree_nodes, dtype=torch.float32).unsqueeze(0).to(device) 
        env_map_tensor = torch.tensor(self.env_map, dtype=torch.float32).unsqueeze(0).to(device) 
        with torch.no_grad():
            next_sample_point = model(tree_nodes_tensor, env_map_tensor)
        node_next = next_sample_point.cpu().numpy()[0]
        return Node(node_next[0], node_next[1], node_next[2])

    def create_env_map(self):
        env_map = np.zeros(self.map_dim, dtype=np.uint8)
        for (ox, oy, oz, size) in self.obstacle_list:
            for i in range(int(ox - size), int(ox + size)):
                for j in range(int(oy - size), int(oy + size)):
                    for k in range(int(oz - size), int(oz + size)):
                        if 0 <= i < self.map_dim[0] and 0 <= j < self.map_dim[1] and 0 <= k < self.map_dim[2]:
                            if np.linalg.norm((ox - i, oy - j, oz - k)) <= size:
                                env_map[i, j, k] = 1
        return env_map

    def get_nearest_node_index(self, random_node):
        distances = [np.linalg.norm((node.x - random_node.x, node.y - random_node.y, node.z - random_node.z)) for node in self.node_list]
        return distances.index(min(distances))

    def steer(self, from_node, to_node):
        angle_xy = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        angle_z = np.arctan2(to_node.z - from_node.z, np.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2))
        new_x = from_node.x + self.step_size * np.cos(angle_xy) * np.cos(angle_z)
        new_y = from_node.y + self.step_size * np.sin(angle_xy) * np.cos(angle_z)
        new_z = from_node.z + self.step_size * np.sin(angle_z)
        new_node = Node(new_x, new_y, new_z)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm((new_node.x - from_node.x, new_node.y - from_node.y, new_node.z - from_node.z))
        return new_node

    def is_collision(self, node):
        for (ox, oy, oz, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dz = oz - node.z
            if dx**2 + dy**2 + dz**2 <= size**2:
                return True
        return False

    def get_nearby_nodes(self, new_node):
        nearby_nodes = []
        for node in self.node_list:
            if np.linalg.norm((node.x - new_node.x, node.y - new_node.y, node.z - new_node.z)) <= self.search_radius:
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
            if not self.is_collision(temp_node) and new_node.cost + np.linalg.norm((new_node.x - node.x, new_node.y - node.y, new_node.z - node.z)) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + np.linalg.norm((new_node.x - node.x, new_node.y - node.y, new_node.z - node.z))

    def generate_path(self, last_node):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        node = last_node
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def find_shortest_path_in_tree(self):
        min_cost = float('inf')
        best_goal_node = None
        for node in self.node_list:
            if np.linalg.norm((node.x - self.goal.x, node.y - self.goal.y, node.z - self.goal.z)) <= self.step_size:
                if node.cost < min_cost:
                    min_cost = node.cost
                    best_goal_node = node
        if best_goal_node:
            return self.generate_path(best_goal_node)
        else:
            return None

    def plan_mixed(self, model, device, alpha=0.5):
        path_len_list = []
        refine_path_len_list = []
        t1 = time.time()
        initial_path_found = False
        initial_path = None
        iter = 0
        for k in range(self.max_iter):
            if np.random.rand() > alpha:
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
                if np.linalg.norm((new_node.x - self.goal.x, new_node.y - self.goal.y, new_node.z - self.goal.z)) <= self.step_size:
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
            if np.random.rand() > alpha:
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
            if current_path and current_path_len < cost_min:
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

    def plot_path(self, path, title="RRTStar 3D Path with Obstacles"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (ox, oy, oz, radius) in self.obstacle_list:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = ox + radius * np.cos(u) * np.sin(v)
            y = oy + radius * np.sin(u) * np.sin(v)
            z = oz + radius * np.cos(v)
            ax.plot_surface(x, y, z, color='red', alpha=0.3)
        tree_nodes = self.node_list
        tree_nodes_x = [node.x for node in tree_nodes]
        tree_nodes_y = [node.y for node in tree_nodes]
        tree_nodes_z = [node.z for node in tree_nodes]
        ax.scatter(tree_nodes_x, tree_nodes_y, tree_nodes_z, c='blue', label='RRT* Nodes')
        if path is not None:
            path_x = [node[0] for node in path]
            path_y = [node[1] for node in path]
            path_z = [node[2] for node in path]
            ax.plot(path_x, path_y, path_z, c='green', linewidth=2, label='Optimized Path')
        ax.scatter(self.start.x, self.start.y, self.start.z, c='green', label='Start')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='yellow', label='Goal')
        ax.set_xlim(0, self.map_dim[0])
        ax.set_ylim(0, self.map_dim[1])
        ax.set_zlim(0, self.map_dim[2])
        ax.legend()
        ax.set_title(title)
        plt.grid(True)
        plt.show()
        plt.close()