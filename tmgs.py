import time
import cvxpy as cp
from scipy.optimize import linprog
import numpy as np
import scipy.sparse as sp


def getA_k(G, k):
    G_k = np.linalg.matrix_power(G, k)
    A_k = np.where(G_k != 0, 1, 0)
    return A_k
def fastest_mixing(G):

    n = G.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    constraints = []
    constraints += [cp.sum(P, axis=0) == np.ones(n)]
    if np.any(G == 0):
        constraints += [P[G == 0] == 0]
    constraints +=  [P[G == 1] >= 1e-5]
    J = np.ones((n, n))
    objective = cp.Minimize(cp.norm(P - (1 / n) * J, 2))
    prob = cp.Problem(objective, constraints)
    scs_params = {
        'max_iters': 10000,
        'eps': 1e-4,
        'acceleration_lookback': 10,
        'scale': 1.0,
    }
    prob.solve(verbose=False, **scs_params)
    P_value = P.value
    P_value[G == 0] = 0

    return P_value, prob.value
def TMGS2(G):
    G2 = getA_k(G, 2)
    Q2, _ = fastest_mixing(G2)
    n = G.shape[0]


    nonzero_vars = []
    var_map = {}  # (l,j,i) -> 变量索引
    idx = 0


    for l in range(n):
        for j in np.where(G[l] == 1)[0]:

            for i in np.where(G[l] == 1)[0]:
                if (l, j, i) not in var_map:
                    var_map[(l, j, i)] = idx
                    nonzero_vars.append((l, j, i))
                    idx += 1
    num_vars = len(nonzero_vars)
    b_eq = []
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0


    for i in range(n):
        for j in np.where(G2[i] == 1)[0]:

            row_entries = []
            for l in np.where(G[i] == 1)[0]:
                if (l, j, i) in var_map:
                    var_idx = var_map[(l, j, i)]
                    row_entries.append(var_idx)


            for var_idx in row_entries:
                row_indices.append(constraint_idx)
                col_indices.append(var_idx)
                data.append(1.0)

            b_eq.append(Q2[i, j])
            constraint_idx += 1

    A_eq = sp.csr_matrix((data, (row_indices, col_indices)),
                         shape=(constraint_idx, num_vars))
    b_eq = np.array(b_eq)

    c = np.zeros(num_vars)


    res = linprog(c, A_eq=A_eq, b_eq=b_eq,bounds = [(None, None)] * num_vars, method='highs')

    if not res.success:
        raise RuntimeError(f"求解失败: {res.message}")


    A = np.zeros((n, n, n))
    for (l, j, i), var_idx in var_map.items():
        A[l, j, i] = res.x[var_idx]

    return A, Q2



def mid_optimization(b_k_1, G, G_k):
    n = G.shape[0]
    A_idx = 0
    A_var_map = {}
    for i in range(n):
        for l in np.where(G[:, i] == 1)[0]:
            for j in np.where(G[:, l] == 1)[0]:
                if (l, j, i) not in A_var_map:
                    A_var_map[(l, j, i)] = A_idx
                    A_idx += 1
    A_k_1 = cp.Variable(len(A_var_map))

    # q_k 的非零位置：(i,j) ∈ E_k 且 i ≤ j（上三角）
    q_var_map = {}
    q_idx = 0
    for i in range(n):
        for j in np.where(G_k[i] == 1)[0]:
            if i <= j:  # 仅存储上三角部分
                q_var_map[(i, j)] = q_idx
                q_idx += 1
    q_k = cp.Variable(len(q_var_map))

    # ================================================
    # 2. 构建稀疏约束条件
    # ================================================
    constraints = []

    # 约束1: 列和等于1（利用对称性）
    col_sums = {j: [] for j in range(n)}
    for (i, j), idx in q_var_map.items():
        col_sums[j].append(idx)
        if i != j:  # 对称位置的变量
            col_sums[i].append(idx)
    for j in col_sums:
        if col_sums[j]:
            constraints.append(cp.sum(q_k[col_sums[j]]) == 1)



    constraints.append(q_k >= 0)

    # 约束3: q_k 的定义式（对称性自动满足）
    for (i, j), q_idx in q_var_map.items():
        if i != j:
            coeffs = []
            A_indices = []
            for l in np.where(G[:, j] == 1)[0]:
                for v_k_2 in np.where(G[:, l] == 1)[0]:
                    if (l, v_k_2, j) in A_var_map:
                        coeffs.append(b_k_1[v_k_2, i, l])
                        A_indices.append(A_var_map[(l, v_k_2, j)])
            if coeffs:
                constraints.append(q_k[q_idx] == cp.sum(cp.multiply(A_k_1[A_indices], coeffs)))
        coeffs = []
        A_indices = []
        for l in np.where(G[:, i] == 1)[0]:
            for v_k_2 in np.where(G[:, l] == 1)[0]:
                if (l, v_k_2, i) in A_var_map:
                    coeffs.append(b_k_1[v_k_2, j, l])
                    A_indices.append(A_var_map[(l, v_k_2, i)])
        if coeffs:
            constraints.append(q_k[q_idx] == cp.sum(cp.multiply(A_k_1[A_indices], coeffs)))


    Q_k = cp.Variable((n, n),symmetric = True)
    for i in range(n):
        for j in range(i,n):
            if G_k[i,j] == 0:
                constraints.append(Q_k[i,j] == 0)
            else:
                constraints.append(Q_k[i,j] == q_k[q_var_map[(i,j)]])


    J = (1 / n) * np.ones((n, n))
    objective = cp.Minimize(cp.norm(Q_k - J, 2))



    problem = cp.Problem(objective, constraints)
    scs_params = {
        'max_iters': 10000,
        'eps': 1e-4,  # 宽松精度容忍
        'acceleration_lookback': 10,
        'scale': 1.0,
    }
    problem.solve(solver=cp.SCS, verbose=False,**scs_params)



    try:

        # 重建对称矩阵
        Q_k = np.zeros((n, n))
        for (i, j), idx in q_var_map.items():
            Q_k[i, j] = q_k.value[idx]
            if i != j:
                Q_k[j, i] = q_k.value[idx]  # 对称赋值

        # 重建 A_k_1
        A_k_1_dense = np.zeros((n, n, n))
        for (l, j, i), idx in A_var_map.items():
            A_k_1_dense[l, j, i] = A_k_1.value[idx]

        return A_k_1_dense, Q_k, problem.value
    except:
        raise ValueError('求解失败')


def TMGSK(G, K=3, verbose = True,rate=2):

    # Input params:
        # G: 标准邻接矩阵 ndarray or 2d-list
        # K: gossip steps
    assert K >= 2 # 这里仅考虑 K>=2 的情况
    n = len(G)
    start = time.time()
    A_1, Q2 = TMGS2(G)
    print('k = 2')
    print(f'costing:{time.time() - start }s')
    A = [A_1]
    q = [Q2]
    Q = [Q2]
    b_k_2 = A_1

    for k in range(2, K):

        G_k = getA_k(G, k + 1)
        Q_k, mu = fastest_mixing(G_k) # ideal case
        start = time.time()
        A_k_1, q_k, mixing_rate = mid_optimization(b_k_2, G, G_k)  # 20式迭代
        if verbose:
            print(f'k = {k + 1}')
            print(f'Q_{k + 1}:{round(mu, 6)},q_{k + 1}:{round(mixing_rate, 6)},costing:{time.time() - start}s')
            # 看 K = k的情况下 每一步的mixing rate 和 ideal case的 mixing rate的差距
        temp = np.zeros([n, n, n])
        for i in range(n):
            for l in range(n):
                mask = G[:, l] == 1
                temp[l, :, i] = np.dot(A_k_1[l, mask, i], b_k_2[mask, :, l])
        b_k_2 = temp

        q.append(q_k)
        A.append(A_k_1)
        Q.append(Q_k)
    # Output params:
    # A: List[3d Tensor] 存每一步的Tensor矩阵 [A1,A2,...Ak-1]
    # q: List[2d array] 存每一步优化的最优的邻接矩阵(practice) [q2,q3,...qk]
    # Q: List[2d array] 存每一步的ideal 邻接矩阵 [Q2,Q3,...,Qk]

    return A, q, Q

