import heapq
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')

    def __lt__(self, other):
        return self.f < other.f

class AStarDatasetGenerator:
    def __init__(self, start, goal, map_dim, obstacle_list, step_scale=0.03, goal_sample_rate=0.1, max_iter=10000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_dim = map_dim
        self.obstacle_list = obstacle_list
        self.step_size = int(np.sqrt(map_dim[0]**2+map_dim[1]**2)*step_scale)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.dataset_path = []
        self.env_map = self.create_env_map()

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            return Node(np.random.uniform(0, self.map_dim[0]), np.random.uniform(0, self.map_dim[1]))
        else:
            return Node(self.goal.x, self.goal.y)

    def create_env_map(self):
        env_map = np.zeros(self.map_dim, dtype=np.uint8)
        for (ox, oy, size) in self.obstacle_list:
            for i in range(int(ox - size), int(ox + size)):
                for j in range(int(oy - size), int(oy + size)):
                    if 0 <= i < self.map_dim[0] and 0 <= j < self.map_dim[1]:
                        if np.linalg.norm((ox - i, oy - j)) <= size:
                            env_map[i, j] = 1

        return env_map

    def heuristic(self, node):      
        return abs(node.x - self.goal.x) + abs(node.y - self.goal.y)

    def get_neighbors(self, node):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),(1, 1),(-1, 1),(1, -1),(-1, -1)]  
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if 0 <= nx < self.map_dim[0] and 0 <= ny < self.map_dim[1] and not self.is_collision(Node(nx, ny)):
                neighbors.append(Node(nx, ny))
        return neighbors

    def is_collision(self, node):
        for (ox, oy, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            if dx**2 + dy**2 <= size**2+25:
                return True
        return False

    def simplify_path(self, path): 
        simplified_path = [path[0]]
        for i in range(2, len(path)): 
            prev_point = simplified_path[-1] 
            current_point = path[i] 
            if np.linalg.norm(np.array(prev_point) - np.array(current_point)) > self.step_size: 
                simplified_path.append(current_point) 
        simplified_path.append(path[-1]) 
        return simplified_path

    def plan(self):
        open_list = []
        closed_list = set()
        self.start.g = 0
        self.start.h = self.heuristic(self.start)
        self.start.f = self.start.g + self.start.h
        heapq.heappush(open_list, self.start)
        cnt = 0
        while open_list and cnt<self.max_iter:
            cnt+=1
            current_node = heapq.heappop(open_list)
            if (current_node.x, current_node.y) == (self.goal.x, self.goal.y):
                path = self.reconstruct_path(current_node)
                sim_path = self.simplify_path(path)
                self.build_path_dataset(sim_path)
                return sim_path
            closed_list.add((current_node.x, current_node.y))
            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y) in closed_list:
                    continue
                tentative_g = current_node.g + np.linalg.norm([current_node.x - neighbor.x, current_node.y - neighbor.y])
                if tentative_g < neighbor.g:
                    neighbor.parent = current_node
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(open_list, neighbor)

        return None

    def plot_path(self,path,title="Astar Path with Obstacles"):
        plt.figure(figsize=(8, 8))
        for (ox, oy, radius) in self.obstacle_list:
            circle = plt.Circle((ox, oy), radius, color='red')
            plt.gca().add_patch(circle)
        if path != None:
            tree_nodes = path
            tree_nodes_x = list()
            tree_nodes_y = list()
            for node in tree_nodes:
                tree_nodes_x.append(node[0])
                tree_nodes_y.append(node[1])
                plt.scatter(node[0],node[1],c='blue')
            plt.plot(tree_nodes_x, tree_nodes_y, c='green', linewidth=2, label='Optimized Path')
        plt.scatter(self.start.x, self.start.y, c='green', label='Start')
        plt.scatter(self.goal.x, self.goal.y, c='red', label='Goal')
        plt.xlim(0, self.map_dim[0])
        plt.ylim(0, self.map_dim[1])
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.savefig(title + '.png')


    def build_path_dataset(self, path):
        for i in range(1, len(path)):
            node = path[i]
            self.dataset_path.append((path[:i], path[i], self.env_map.copy()))
        return
