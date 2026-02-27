from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np



def compute_otloss(pi, C_prime, epsilon, lambda_val, a):
    """优化后的损失计算，直接使用pi"""
    # 运输成本
    transport = np.sum(pi * C_prime)
    
    # 熵项 
    entropy = -epsilon * np.sum(pi * (np.log(pi) - 1))

    # KL散度
    pi_marginal = np.sum(pi, axis=1)  # 使用pi的行和来计算边际分布

    kl = lambda_val * (np.dot(pi_marginal, np.log(pi_marginal / a)) 
                       - np.sum(pi_marginal) + np.sum(a))

    return transport, entropy, kl, transport + entropy + kl

def compute_Dualloss(f, K, g, a, b, epsilon, lambda_val):

    term1 = -epsilon * np.dot(f, K @ g)

    term2 = -lambda_val * np.sum(a * (f ** (-epsilon / lambda_val)))

    term3 = lambda_val * np.sum(a) + epsilon * np.sum(np.log(g) * b)

    return term1 + term2 + term3


def compute_term0(D, D_prime, pi):

    n = D.shape[0]
    m = D_prime.shape[0]
    # 计算行和
    row_sums = np.sum(pi, axis=1).reshape(-1, 1)

    # 计算列和
    col_sums = np.sum(pi, axis=0).reshape(-1, 1)

    result1 = (D**2) @ row_sums @ (np.ones((1, m)))

    result2 = (np.ones((n, 1))) @ ((D_prime**2) @ col_sums).T
   
    result3 = D @ pi @ D_prime.T

    return result1 + result2 - 2*result3


def compute_triplet(D, D_prime, pi):
    """
    计算代价矩阵中与三元组相关的β项
    
    参数:
        D: 原始距离矩阵 (n x n)
        D_prime: 目标距离矩阵 (m x m)
        pi: 当前传输矩阵 (n x m)
        
    返回:
        triplet_term: 三元组项的梯度贡献 (n x m矩阵)
    """
    # 第一项: sum_k π_ik D'_jk^2 → 矩阵乘法实现
    term1 = pi @ (D_prime ** 2).T
    
    # 第二项: sum_i π_iq D_ip^2 → 利用广播和转置
    term2 = (D ** 2) @ pi
    
    return term1, term2

def normalize_M(M, C, N1, _ord='fro'):
    '''
    A function to return normalization constants for the gradient involving the merged feature-spatial matrix,
    which is normalized by a factor related to the interslice feature cost C.
    ------Parameters------
    M: torch.tensor (N x N)
        A symmetric matrix with positive entries (i.e. distance matrix, merged feature-spatial matrix)
    C: torch.tensor (N x M) or (M x N)
        Matrix of pairwise feature distances between slice 1 and 2
    N1: int
        Number of spots in first slice
    '''
    return (np.linalg.norm(C, ord=_ord)**(1/2) / np.linalg.norm(M, ord=_ord)) * (N1 )**(1/2)
    
def fused_gw_adaptive_ot(C, D, D_prime, alpha, epsilon, lambda_val, beta, max_outer=100, max_middle=50, max_inner=1000, tol=1e-6, sinkhorn_tol=1e-6):
    """融合Gromov-Wasserstein的自适应最优传输算法（三重循环结构）"""
    n, m = C.shape
    b = np.ones(m) / m
    a = np.ones(n) / n  # 初始分布
    pi = np.outer(a, b)  # 初始耦合矩阵
    stats, a_history,pi_history = [], [], []
    prev_total_loss = None
    
    for outer_iter in range(max_outer):
        
        
        middle_loss_history = []

        middle_entropy_history = []

        middle_transport_history = []

        middle_kl_history = []

        # ================== 中间层循环 ==================
        prev_middle_loss = None

        p1, p2 = normalize_M(D, C, n, _ord='fro'), \
                                    normalize_M(D_prime, C, n, _ord='fro')

        D_nor = p1 *  D
        D_prime_nor = p2 * D_prime

        term1 = compute_term0(D_nor, D_prime_nor, pi) 

        print(term1.mean(),C.mean())

        r2, r1 = np.linalg.norm(C, ord='fro')/np.linalg.norm(D_prime**2, ord='fro') * (n**1/2), \
                        np.linalg.norm(C, ord='fro')/np.linalg.norm(D**2, ord='fro') * (n / m**(1/2))

        triplet_term1, triplet_term2= compute_triplet(D, D_prime, pi) 

        term2 = r1*(triplet_term1) + r2*(triplet_term2)

        #C_prime = alpha * term1 + (1 - alpha) * C
        
        
        C_prime = (1-alpha)*C + alpha*(beta*term1 + (1-beta)*term2)
        print(C_prime.mean())

        # 2. 更新核矩阵 K
        log_K = -C_prime / epsilon
        K = np.exp(log_K)
        K_T = K.T.copy()

        for middle_iter in range(max_middle):
            
            print(middle_iter)
    

            f = np.ones(n)
            g = np.ones(m)
            # --- 内层Sinkhorn迭代  ---

            #dueloss = []
            for inner_step in range(1000):

                prev_f, prev_g = f.copy(), g.copy()

                # 更新f
                K_dot_g = K @ g

                np.power(a / K_dot_g, lambda_val/(lambda_val+epsilon), out=f)
                
                # 更新g
                K_T_dot_f = K_T @ f
                np.divide(b, K_T_dot_f, out=g)
                
                # 收敛判断
                #f_diff = np.linalg.norm(f - prev_f)
                #g_diff = np.linalg.norm(g - prev_g)

                current_Dualloss = compute_Dualloss(f, K, g, a, b, epsilon, lambda_val)

                #dueloss.append(current_Dualloss)

                # 首次迭代初始化prev_Dualloss
                if inner_step == 0:
                    prev_Dualloss = current_Dualloss
                    continue  # 跳过第一次的差异计算

                # 计算三要素差异
                dual_diff = current_Dualloss - prev_Dualloss
                prev_Dualloss = current_Dualloss
                  
                # 收敛条件
                if (
                    abs(dual_diff) < sinkhorn_tol):
                    break

            # 3. 更新耦合矩阵π
            pi = f[:, None] * K * g[None, :]
            #term1 = compute_term0(D, D_prime, pi)
            #C_prime = alpha * term1 + (1 - alpha) * C
            # 


            transport, entropy, kl, midd_current_loss = compute_otloss(pi, C_prime, epsilon, lambda_val, a)

            # # 1. 更新边缘分布a
            pi_marginal = np.sum(pi, axis=1)
            a = pi_marginal/ pi_marginal.sum()

            middle_loss_history.append(midd_current_loss)
            middle_entropy_history.append(entropy)
            middle_transport_history.append(transport)
            middle_kl_history.append(kl)


            # 中间层收敛判断（基于损失变化量）
            if middle_iter > 0:
                loss_diff = abs(midd_current_loss - prev_middle_loss)
                if loss_diff < tol:
                    print(f"中间层迭代 {middle_iter+1}/{max_middle}")

                    break
            prev_middle_loss = midd_current_loss
        
       
        a_history.append(a)


        # 2. 计算损失
        _, _, _, total_loss = compute_otloss(pi, C_prime, epsilon, lambda_val, a)
        # 3. 记录统计量
        stats.append({
            "outer_iter": outer_iter,
            "middle_iter": middle_iter,
            "total_loss": total_loss,
            "transport_cost": middle_transport_history,
            "entropy": middle_entropy_history,
            "kl": middle_kl_history,
            "middle_loss": middle_loss_history
        })
        

        # 4. 外层收敛判断
        if outer_iter > 0 and abs(prev_total_loss - total_loss) < tol:
            break
        prev_total_loss = total_loss
        print(f"外层迭代 {outer_iter+1}/{max_outer}")
    
    return a, pi, stats, a_history