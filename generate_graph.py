import numpy as np
from functools import partial
import networkx as nx
import math

def circle(i, d=1, M=10):
    """
    生成一个环形网络的邻接矩阵的一行。

    参数：
    - i: 当前节点的索引。
    - d: 节点的度数，即每个节点连接的邻居节点数量。
    - M: 节点总数。

    返回值：
    - W: 环形网络的邻接矩阵的一行。
    """
    W = np.zeros([M,])
    idx = np.arange(i + 1, i + 1 + d) % M
    W[idx] = 1
    return W

def generate_ring(n_agent=20, D=2):
    """
    生成一个环形网络的邻接矩阵。

    参数：
    - n_client: 节点总数。
    - D: 节点的度数，即每个节点连接的邻居节点数量。

    返回值：
    - W_symmetric: 环形网络的邻接矩阵。
    """
    pargenerateW3 = partial(circle, d=D, M=n_agent)
    W = np.apply_along_axis(pargenerateW3, 1, np.arange(n_agent).reshape([-1, 1]))
    W_symmetric = W + W.T
    np.fill_diagonal(W_symmetric, 1)
    return np.array(W_symmetric)

def generate_fixed_degree(n_agent=20, D=3):
    """
    生成一个固定度数的无向网络的邻接矩阵。

    参数：
    - n: 节点总数。
    - D: 每个节点的固定度数（连接的节点数量）。

    返回值：
    - adjacency_matrix: 固定度数网络的邻接矩阵。

    注意：
    - D 必须小于 n。
    - n * D 必须是偶数以保证无向图的每个节点度数固定。
    """

    if D >= n_agent:
        raise ValueError("D 必须小于 n_agent")
    if n_agent * D % 2 != 0:
        raise ValueError("n * D 必须是偶数以保证无向图的每个节点度数固定")


    tries = 0
    while tries < 10:
        adjacency_matrix = np.zeros((n_agent, n_agent), dtype=int)
        edges = [(i, j) for i in range(n_agent) for j in range(i + 1, n_agent)]
        while edges:
            edge = edges.pop(np.random.randint(len(edges)))
            i, j = edge
            if adjacency_matrix[i].sum() < D and adjacency_matrix[j].sum() < D:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
        np.fill_diagonal(adjacency_matrix, 1)
        if nx.is_connected(nx.from_numpy_array(adjacency_matrix)):
            break
        else:
            print('not connected,retrying...')
            tries += 1
    if tries == 5:
        raise ValueError("G is not connected")
    return adjacency_matrix

def generate_grid(n_agent = 20, n=4):
    """
    生成一个 m x n 的网格网络的邻接矩阵。

    参数：
    - m: 网格的行数。
    - n: 网格的列数。

    返回值：
    - adjmat: 网格网络的邻接矩阵。
    """
    m = n_agent // n
    G = nx.grid_2d_graph(m, n)
    adjmat = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(adjmat, 1)
    return np.array(adjmat)

def generate_torus(n_agent = 20 , n=4):
    """
    生成一个 m x n 的环形网格网络的邻接矩阵。

    参数：
    - m: 环形网格的行数。
    - n: 环形网格的列数。

    返回值：
    - adjmat: 环形网格网络的邻接矩阵。
    """
    m = n_agent // n
    G = nx.grid_2d_graph(m, n, periodic=True)
    adjmat = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(adjmat, 1)
    return np.array(adjmat)

def generate_complete_binary_tree(n_agent):
    """
    生成一个包含给定节点总数的完全二叉树的邻接矩阵。

    参数：
    - n_site: 完全二叉树的节点总数。

    返回值：
    - adjmat: 完全二叉树的邻接矩阵。
    """
    level = math.floor(math.log2(n_agent))
    G = nx.balanced_tree(2, level)  # 构建一个完美二叉树
    G.remove_nodes_from(range(n_agent, 2 ** (level + 1) - 1))
    adjmat = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(adjmat, 1)
    return np.array(adjmat)

def generate_ER(n_agent, pi=0.5):
    """
    生成一个包含给定节点总数和连边概率的 ER（Erdős-Rényi）随机图的邻接矩阵。

    参数：
    - n_site: 节点总数。
    - pi: 连边概率。

    返回值：
    - adjmat: ER 随机图的邻接矩阵。
    """
    G = nx.erdos_renyi_graph(n_agent, pi)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_agent, pi)
    adjmat = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(adjmat, 1)
    return np.array(adjmat)
