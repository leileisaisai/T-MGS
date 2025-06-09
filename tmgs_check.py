atol = 1e-3
import numpy as np
from tmgs import getA_k

def check_solution(A, G):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            for l in range(n):
                if G[i, j] == 0 or G[i, l] == 0:
                    if np.abs(A[i, j, l]) > atol:
                        print(f"A[{i}, {j}, {l}] should be 0 but is {A[i, j, l]}")
                        return False
    return True


def check_q_constraints(q, G):
    n = q.shape[0]
    # 检查q是否对称
    if not np.allclose(q, q.T):
        print("Matrix q is not symmetric")
        return False

    # 检查q是否非负
    if np.any(q < -atol):
        print()
        print(f"Matrix q has negative elements: {np.min(q)}")
        return False

    # 检查q的每列之和是否为1
    if not np.allclose(np.sum(q, axis=0), np.ones(n), atol=atol):
        print("The sum of the columns in matrix q is not equal to 1")
        return False

    # 检查如果G[i, j] == 0，则q[i, j]也应该为0
    for i in range(n):
        for j in range(n):
            if G[i, j] == 0 and not np.isclose(q[i, j], 0, atol=atol):
                print(f"q[{i}, {j}] should be 0 but is {q[i, j]}")
                return False
    return True


def check_15(q, A_list, G):
    n = q.shape[0]
    K = len(A_list) + 1

    for i in range(n):
        for j in range(n):
            sum_paths = 0
            paths = generate_paths(G, K, j, i)  # 生成所有K步路径

            for path in paths:
                product = 1
                for k in range(K - 1):
                    product *= A_list[k][path[k + 1], path[k], path[k + 2]]
                sum_paths += product

            if not np.isclose(q[i, j], sum_paths, atol=atol):
                print(f"q[{i}, {j}] does not satisfy the constraint: q[{i}, {j}]:{q[i, j]}, sum_path {sum_paths}")
                return False
    return True


def generate_paths(G, K, start, end):
    n = G.shape[0]
    paths = []

    def dfs(current, steps, path):
        if steps == K:
            if current == end:
                paths.append(path[:])
            return

        for next_node in range(n):
            if G[current, next_node] == 1:  # 确保节点间有边连接
                path.append(next_node)
                dfs(next_node, steps + 1, path)
                path.pop()

    dfs(start, 0, [start])
    return paths


def check_q_k_A_k(q, A_list, G):
    k = len(q) + 1
    A_list = np.array(A_list)
    for i in range(k - 2):
        temp_q = q[i + 1]

        if not check_15(temp_q, A_list[:i + 2], G):
            return False
    return True


def check_q_k(q, G):
    k = len(q)
    for i in range(k):
        temp_q = q[i]
        G_k = getA_k(G, i + 2)
        if not check_q_constraints(temp_q, G_k):
            print(i)
            return False
    return True


def check_A(A_list, G):
    k = len(A_list)
    for i in range(k):

        if not check_solution(A_list[i], G):
            print(i)
            return False
    return True