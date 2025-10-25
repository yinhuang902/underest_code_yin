# 修正后的 minimize_on_each_triangle_multi 函数
# 主要修复：将ms聚合计算移到场景循环外部

def minimize_on_each_triangle_multi_fixed(
    nodes: List[np.ndarray],
    scenarios: List[Dict],
    weights: np.ndarray,
    agg: AggT = AGG_DEFAULT,
    alpha: float = CVaR_ALPHA_DEF,
    solver: str = 'gurobi',
    tee: bool = False
):
    """
    修正版本：修复ms聚合计算位置错误
    """
    pts = np.asarray(nodes, float)
    tri = Delaunay(pts)

    # Pyomo model reused across triangles/scenarios
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 10.0))
    m.y = pyo.Var(bounds=(0.0, 10.0))
    m.lam = pyo.Var(range(3), domain=pyo.NonNegativeReals)
    m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in range(3)) == 1.0)
    m.x_link_x = pyo.Constraint(expr=m.x == 0.0)  # placeholder
    m.x_link_y = pyo.Constraint(expr=m.y == 0.0)  # placeholder
    m.As = pyo.Var()

    opt = pyo.SolverFactory(solver)
    if USE_GUROBI_OPT and solver.lower().startswith('gurobi'):
        opt.options.update({
            'OutputFlag': 0,
            'OptimalityTol': 1e-9,
            'BarConvTol': 1e-8,
            'NonConvex': 2,
        })

    results = []
    S = len(scenarios)
    w = np.asarray(weights, float)
    w = w / w.sum()

    for k, simp in enumerate(tri.simplices):
        verts = pts[simp]  # (3,2)

        # bind x,y to barycentric coords of this triangle
        m.del_component(m.x_link_x); m.del_component(m.x_link_y)
        m.x_link_x = pyo.Constraint(expr=m.x == sum(m.lam[j]*float(verts[j,0]) for j in range(3)))
        m.x_link_y = pyo.Constraint(expr=m.y == sum(m.lam[j]*float(verts[j,1]) for j in range(3)))

        Fverts = fverts_per_scenario(verts, scenarios)  # (S,3)

        ms_s = np.zeros(S, dtype=float)
        x_ms_s = np.zeros((S,2), dtype=float)
        f_at_x_ms_s = np.zeros(S, dtype=float)

        # 第一步：计算所有场景的 ms_s
        for s, theta in enumerate(scenarios):
            # As_s = sum λ_j f_s(v_j)
            if hasattr(m, 'As_con'):
                m.del_component(m.As_con)
            m.As_con = pyo.Constraint(expr=m.As == sum(m.lam[j]*float(Fverts[s,j]) for j in range(3)))

            # objective: min f_s(x,y) - As_s
            if hasattr(m, 'obj'):
                m.del_component(m.obj)
            f_expr_s = scenario_quadratic_expr(m, theta)
            m.obj = pyo.Objective(expr=f_expr_s - m.As, sense=pyo.minimize)

            res = opt.solve(m, tee=tee)

            x_ms_s[s,:] = [pyo.value(m.x), pyo.value(m.y)]
            ms_s[s] = float(pyo.value(m.obj))
            f_at_x_ms_s[s] = float(scenario_quadratic_numpy(theta, x_ms_s[s,:]))

        # 第二步：在所有场景计算完成后进行聚合
        if agg == 'mean':
            ms_agg = float(np.sum(w*ms_s))
        elif agg == 'max':
            ms_agg = float(np.max(ms_s))
        elif agg == 'cvar':
            ms_agg = float(weighted_cvar(ms_s, w, alpha))
        else:
            raise ValueError("agg must be 'mean'|'max'|'cvar'")

        # 统计信息
        ms_mean = float(np.sum(w*ms_s))
        ms_max  = float(np.max(ms_s))
        ms_p90  = float(weighted_quantile(ms_s, w, 0.9))
        ms_std  = float(np.sqrt(np.sum(w*(ms_s - ms_mean)**2)))

        # 计算下界 LB
        LB_s = np.min(Fverts, axis=1) + ms_s  # 形状 (S,)
        if agg == 'mean':
            LB_agg = float(np.sum(w * LB_s))
        elif agg == 'max':
            LB_agg = float(np.max(LB_s))
        else:  # cvar
            LB_agg = float(weighted_cvar(LB_s, w, alpha))

        # 计算上界 UB
        Gv = np.zeros(3, dtype=float)
        for j in range(3):
            vals = Fverts[:, j]
            if agg == 'mean':
                Gv[j] = float(np.sum(w * vals))
            elif agg == 'max':
                Gv[j] = float(np.max(vals))
            else:  # cvar
                Gv[j] = float(weighted_cvar(vals, w, alpha))

        UB_verts = float(np.min(Gv))

        # f(x_ms)的聚合值
        if agg == 'mean':
            f_x_ms_agg = float(np.sum(w * f_at_x_ms_s))
        elif agg == 'max':
            f_x_ms_agg = float(np.max(f_at_x_ms_s))
        else:
            f_x_ms_agg = float(weighted_cvar(f_at_x_ms_s, w, alpha))

        UB = min(UB_verts, f_x_ms_agg)

        # 根据算法描述，找到ms最小的场景对应的点
        min_ms_idx = int(np.argmin(ms_s))
        x_ms_best = x_ms_s[min_ms_idx]

        # 计算所有场景x_ms的平均值（原代码使用）
        x_ms_mean = np.mean(x_ms_s, axis=0)

        results.append({
            "simplex_index": k,
            "ms_agg": ms_agg,
            "ms_mean": ms_mean,
            "ms_max": ms_max,
            "ms_p90": ms_p90,
            "ms_std": ms_std,
            "ms_raw": ms_s.copy(),
            "LB": LB_agg,
            "UB": UB,
            "x_ms_mean": x_ms_mean,
            "x_ms_best": x_ms_best,  # 新增：ms最小的场景对应的点
            "x_ms_all": x_ms_s.copy(),
            "active": True  # 将在主循环中更新
        })

    return tri, results


# 修正后的候选点选择函数
def pick_candidate_by_ms_rank_multi_fixed(candidates, nodes, min_dist, kappa=1.0):
    """
    修正版本：可以选择使用x_ms_best（ms最小的点）或x_ms_mean
    """
    nodes_arr = np.asarray(nodes, float)
    
    # 计算每个候选单形的LCB分数
    scores = []
    for r in candidates:
        ms_std = r["ms_std"]
        # 根据算法描述，score = LB - k*sigma
        score = r["LB"] - kappa * ms_std
        scores.append((score, r))
    
    # 按照LCB分数排序（从小到大）
    scores.sort(key=lambda x: x[0])
    
    # 从最佳LCB分数开始尝试
    for rank, (score, r) in enumerate(scores):
        # 可以选择使用 x_ms_best 或 x_ms_mean
        # x_cand = r["x_ms_best"]  # 使用ms最小的点
        x_cand = r["x_ms_mean"]  # 保持原代码逻辑
        
        # 检查是否太接近已有节点
        too_close = any(np.linalg.norm(x_cand - n) < min_dist for n in nodes_arr)
        
        if not too_close:
            return x_cand, rank, score, r["simplex_index"]
    
    return None, -1, float('inf'), -1


# 如果需要使用基于最小ms_i的候选点选择策略
def pick_candidate_by_min_ms_fixed(candidates, nodes, min_dist, kappa=1.0):
    """
    替代版本：直接选择ms_i最小的点作为候选
    """
    nodes_arr = np.asarray(nodes, float)
    
    # 找到所有单形中ms最小的点
    best_ms = float('inf')
    best_cand = None
    best_simplex = -1
    
    for r in candidates:
        ms_raw = r["ms_raw"]
        min_idx = np.argmin(ms_raw)
        min_ms = ms_raw[min_idx]
        
        if min_ms < best_ms:
            x_cand = r["x_ms_all"][min_idx]
            too_close = any(np.linalg.norm(x_cand - n) < min_dist for n in nodes_arr)
            
            if not too_close:
                best_ms = min_ms
                best_cand = x_cand
                best_simplex = r["simplex_index"]
    
    if best_cand is not None:
        return best_cand, 0, best_ms, best_simplex
    
    return None, -1, float('inf'), -1
