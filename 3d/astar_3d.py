import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Node3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.g = float('inf') 
        self.h = float('inf')  
        self.f = float('inf')  
    def __lt__(self, other):
        return self.f < other.f

class AStar3D:
    def __init__(self, start, goal, map_dim, obstacle_list, step_scale=0.03, goal_sample_rate=0.1, max_iter=10000):
        self.start = Node3D(start[0], start[1], start[2])
        self.goal = Node3D(goal[0], goal[1], goal[2])
        self.map_dim = map_dim
        self.obstacle_list = obstacle_list
        self.step_size = int(np.sqrt(map_dim[0]**2 + map_dim[1]**2 + map_dim[2]**2) * step_scale)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.dataset_path = []
        self.env_map = self.create_env_map()

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            return Node3D(np.random.uniform(0, self.map_dim[0]), np.random.uniform(0, self.map_dim[1]), np.random.uniform(0, self.map_dim[2]))
        else:
            return Node3D(self.goal.x, self.goal.y, self.goal.z)

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

    def heuristic(self, node):
        return abs(node.x - self.goal.x) + abs(node.y - self.goal.y) + abs(node.z - self.goal.z)

    def get_neighbors(self, node):
        neighbors = []
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
                      (1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1),
                      (1, -1, -1), (-1, 1, 1), (1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1)]
        for dx, dy, dz in directions:
            nx, ny, nz = node.x + dx, node.y + dy, node.z + dz
            if 0 <= nx < self.map_dim[0] and 0 <= ny < self.map_dim[1] and 0 <= nz < self.map_dim[2] and not self.is_collision(Node3D(nx, ny, nz)):
                neighbors.append(Node3D(nx, ny, nz))
        return neighbors

    def is_collision(self, node):
        for (ox, oy, oz, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dz = oz - node.z
            if dx**2 + dy**2 + dz**2 <= size**2 + 25:
                return True
        return False

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.reverse()
        return path

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
        while open_list and cnt < self.max_iter:
            cnt += 1
            current_node = heapq.heappop(open_list)
            if (current_node.x, current_node.y, current_node.z) == (self.goal.x, self.goal.y, self.goal.z):
                path = self.reconstruct_path(current_node)
                sim_path = self.simplify_path(path)
                self.build_path_dataset(sim_path)
                return sim_path
            closed_list.add((current_node.x, current_node.y, current_node.z))
            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y, neighbor.z) in closed_list:
                    continue
                tentative_g = current_node.g + np.linalg.norm([current_node.x - neighbor.x, current_node.y - neighbor.y, current_node.z - neighbor.z])
                if tentative_g < neighbor.g:
                    neighbor.parent = current_node
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(open_list, neighbor)
        return None

    def plot_path(self, path,title='3D AStar Path Planning'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (ox, oy, oz, size) in self.obstacle_list:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = ox + size * np.outer(np.cos(u), np.sin(v))
            y = oy + size * np.outer(np.sin(u), np.sin(v))
            z = oz + size * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='r', alpha=0.3)
        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g', linewidth=2, label='Path')
        ax.scatter(self.start.x, self.start.y, self.start.z, c='g', marker='o', label='Start')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='r', marker='x', label='Goal')
        ax.set_xlim([0, self.map_dim[0]])
        ax.set_ylim([0, self.map_dim[1]])
        ax.set_zlim([0, self.map_dim[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        plt.savefig(title + '.png')

    def build_path_dataset(self, path):
        for i in range(1, len(path)):
            node = path[i]
            self.dataset_path.append((path[:i], path[i], self.env_map.copy()))
        return
