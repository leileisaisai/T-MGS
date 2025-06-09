from generate_graph import generate_ER,generate_torus,generate_ring,generate_grid,generate_fixed_degree,generate_complete_binary_tree
import numpy as np
from tmgs import TMGS2,TMGSK
from tmgs_check import check_A, check_q_k, check_q_k_A_k
import random
if __name__ == '__main__':

    random.seed(2)
    Graph = {
        f"ER Network": generate_ER(30,0.1),
        f"Ring Network": generate_ring(30, 1),
        f"Fixed Degree Network": generate_fixed_degree(30, 2),
        f"Grid Network ": generate_grid(30,5),
        'Complete Binary Tree Network': generate_complete_binary_tree(30),
        f"Torus Network": generate_torus(30, 5),
    }

    for name, G in Graph.items():
        print(f'==================================')
        print(f'starting {name}')

        A_result, q_result, Q_result = TMGSK(G, K=4)

        # # 检查一下得到的结果是否满足我们的约束
        # # 检查每个A是否满足G的要求
        print(f'checking every A constraint:{check_A(A_result,G)}')
        # 检查每个q是否是双随机
        print(f'checking every q constraint:{check_q_k(q_result,G)}')
        # 检查每个A和q的关系对不对 用DFS验证所有路径相加
        print(f'checking every A via q constraint:{check_q_k_A_k(q_result,A_result,G)}')




    #