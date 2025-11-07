# simplex.py
import numpy as np
from scipy.spatial import Delaunay
from time import perf_counter
from pyomo.opt import SolverStatus, TerminationCondition

from utils import SimplexTracker
from utils import (
    MIN_DIST, ACTIVE_TOL, MS_AGG, MS_CACHE_ENABLE, GAP_STOP_TOL,
    corners_from_var_bounds, evaluate_Q_at, tet_quality, min_dist_to_nodes,
    _print_candidates_table, plot_iteration_plotly, LAST_DEBUG
)
from bundles import ms_on_tetra_for_scene, qmin_on_tetra_for_scene


# ------------------------- Evaluate all tetrahedra (per-scene) -------------------------
def evaluate_all_tetra(nodes, scen_values, ms_bundles, qmin_bundles, first_vars_list,
                       ms_cache=None, cache_on=True, tracker=None):
    """
    计算每个单形在每个场景下的 ms 与 true-Q 最小值，并构建新口径下界：
        LB_old[s]  = min(f_verts_s) + ms_s
        LB_true[s] = max(LB_old[s], qmin_true_s)
        LB (agg)   = sum/mean(LB_true[s] over scenes)
    返回 per_tet 列表，每项包含：
        - ms_per_scene, xms_per_scene
        - qmin_true_scene
        - LB_true_scene, LB (聚合)
        - 其他几何/索引信息
    """
    pts = np.asarray(nodes, dtype=float)
    if len(pts) < 4:
        return None, []
    tri = Delaunay(pts)
    S = len(ms_bundles)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    diam = float(np.linalg.norm(maxs - mins))
    vol_tol = 1e-12 * max(diam**3, 1.0)

    per_tet = []
    for k, simp in enumerate(tri.simplices):
        idxs = list(map(int, simp))
        verts = [tuple(pts[i]) for i in idxs]

        v0, v1, v2, v3 = np.array(verts)
        vol = abs(np.linalg.det(np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)) / 6.0)
        if vol < vol_tol:
            continue

        simplex_id = tuple(sorted(idxs))
        if tracker is not None:
            tracker.note_created(simplex_id)

        # per-vertex true values for each scene
        fverts_per_scene = [[scen_values[ω][i] for i in idxs] for ω in range(S)]
        fverts_sum = [sum(fverts_per_scene[ω][j] for ω in range(S)) for j in range(4)]

        # ===== per-scene ms + qmin_true (with cache) =====
        ms_scene, xms_scene = [], []
        qmin_scene = []

        key_base = simplex_id

        for ω in range(S):
            # ---- ms ----
            key_ms = ("ms", int(ω), key_base)
            hit_ms = cache_on and (ms_cache is not None) and (key_ms in ms_cache)
            if hit_ms:
                ms_val, new_pt = ms_cache[key_ms]
            else:
                ms_val, lam_star, new_pt = ms_on_tetra_for_scene(
                    ms_bundles[ω], verts, fverts_per_scene[ω]
                )
                if cache_on and (ms_cache is not None):
                    ms_cache[key_ms] = (ms_val, new_pt)
                if tracker is not None:
                    tracker.note_ms_recomputed(simplex_id)
            ms_scene.append(float(ms_val))
            xms_scene.append(new_pt)

            # ---- qmin_true ----
            key_qm = ("qmin", int(ω), key_base)
            hit_qm = cache_on and (ms_cache is not None) and (key_qm in ms_cache)
            if hit_qm:
                qmin_val, _pt_true = ms_cache[key_qm]
            else:
                qmin_val, _lam, _pt_true = qmin_on_tetra_for_scene(
                    qmin_bundles[ω], verts
                )
                if cache_on and (ms_cache is not None):
                    ms_cache[key_qm] = (qmin_val, _pt_true)
            qmin_scene.append(float(qmin_val))

        # ---- aggregate ----
        if MS_AGG == "sum":
            ms_total = float(np.sum(ms_scene))
        elif MS_AGG == "mean":
            ms_total = float(np.mean(ms_scene))
        else:
            raise ValueError("MS_AGG must be 'sum' or 'mean'")

        # per-scene lower bounds
        LB_old_scene = [min(fverts_per_scene[ω]) + ms_scene[ω] for ω in range(S)]
        LB_true_scene = [max(LB_old_scene[ω], qmin_scene[ω]) for ω in range(S)]

        if MS_AGG == "sum":
            LB_true = float(np.sum(LB_true_scene))
        else:
            LB_true = float(np.mean(LB_true_scene))

        # legacy UB（保留，不用于 active 判定，只用于展示）
        UB_legacy = float(np.max(fverts_sum) + ms_total)

        best_scene = int(np.argmin(ms_scene))
        x_ms_best = xms_scene[best_scene]

        per_tet.append({
            "simplex_index": k,
            "vert_idx": idxs,
            "verts": verts,
            "volume": vol,

            # metrics
            "ms_per_scene": ms_scene,
            "xms_per_scene": xms_scene,
            "qmin_true_scene": qmin_scene,
            "LB_old_scene": LB_old_scene,
            "LB_true_scene": LB_true_scene,

            # new lower/upper summaries
            "ms": ms_total,
            "LB": LB_true,          # 用新的聚合下界覆盖 LB 字段（主循环直接用它）
            "UB": UB_legacy,

            "x_ms_best_scene": x_ms_best,
            "best_scene": best_scene,
        })

    return tri, per_tet


# ------------------------- MAIN LOOP -------------------------
def run_pid_simplex_3d(base_bundles, ms_bundles, qmin_bundles, model_list, first_vars_list,
                       target_nodes=30, min_dist=MIN_DIST, active_tol=ACTIVE_TOL, verbose=True,
                       agg_bundle=None, gap_stop_tol=GAP_STOP_TOL, tracker: SimplexTracker | None = None):

    if tracker is None:
        tracker = SimplexTracker()

    """
    New implementation summary:
      - Active & global LB are computed from LB_true (see evaluate_all_tetra).
      - Candidate generation仍沿用 ms-based 点（保持你既有选点策略不变）。
    """
    global LAST_DEBUG
    LB_hist, UB_hist, ms_hist, node_count = [], [], [], []
    UB_node_hist, add_node_hist = [], []
    ms_a_hist, ms_b_hist = [], []
    active_ratio_hist = []

    S = len(model_list)
    nodes = corners_from_var_bounds(first_vars_list[0])

    bounds_arr = np.array([[float(v.lb), float(v.ub)] for v in first_vars_list[0]], float)
    diam = float(np.linalg.norm(bounds_arr[:,1] - bounds_arr[:,0]))
    min_dist = float(min_dist)

    # cache f_ω(node_i)
    scen_values = [[None]*len(nodes) for _ in range(S)]
    for i, node in enumerate(nodes):
        for ω in range(S):
            scen_values[ω][i] = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], node)

    it = 0
    stop_due_to_collision = False
    ms_cache = {}   # keys: ("ms"/"qmin", scene_idx, simplex_id) -> (value, pt)
    while len(nodes) < target_nodes:
        t_iter0 = perf_counter()
        tracker.start_iter(it)

        # 1) Global UB (by sum target over nodes)
        f_sum_per_node = [
            sum(scen_values[ω][i] for ω in range(S))
            for i in range(len(nodes))
        ]
        ub_idx = int(np.argmin(f_sum_per_node))
        UB_global = float(f_sum_per_node[ub_idx])
        UB_node = tuple(nodes[ub_idx])

        # 2) Evaluate all tetrahedra under new scheme
        tri, per_tet = evaluate_all_tetra(
            nodes, scen_values, ms_bundles, qmin_bundles, first_vars_list,
            ms_cache=ms_cache, cache_on=True, tracker=tracker
        )
        if tri is None or not per_tet:
            if verbose:
                print("Not enough nodes to make tetrahedra; stop.")
            tracker.end_iter()
            break

        # 3) active mask (LB_true vs UB + tol) + shape quality
        active_mask = {r["simplex_index"]: (r["LB"] <= UB_global + active_tol) for r in per_tet}
        q_cut = 1e-3
        for r in per_tet:
            sid = r["simplex_index"]
            if not active_mask.get(sid, False):
                continue
            q = tet_quality(r["verts"])
            if q < q_cut:
                active_mask[sid] = False

        # 4) active ratio & tracker
        total_vol = sum(r["volume"] for r in per_tet)
        active_vol = sum(r["volume"] for r in per_tet if active_mask[r["simplex_index"]])
        active_ratio = active_vol / total_vol if total_vol > 0 else 0.0

        for r in per_tet:
            if not active_mask.get(r["simplex_index"], False):
                continue
            simplex_id = tuple(sorted(r["vert_idx"]))
            has_ub = (ub_idx in r["vert_idx"])
            tracker.note_active(simplex_id, has_ub=has_ub)
        tracker.end_iter()

        # 5) Global LB from UB-neighborhood actives (delta = LB - UB)
        ub_active = [r for r in per_tet if (ub_idx in r["vert_idx"]) and active_mask.get(r["simplex_index"], False)]
        if ub_active:
            rec_b = min(ub_active, key=lambda r: (r["LB"] - UB_global))
            delta_b = float(rec_b["LB"] - UB_global)
            ms_b_simp = int(rec_b["simplex_index"])
            LB_global = UB_global + delta_b
        else:
            rec_b = None
            ms_b_simp = None
            act_LBs = [r["LB"] for r in per_tet if active_mask.get(r["simplex_index"], False)]
            LB_global = float(min(act_LBs)) if act_LBs else float(min(r["LB"] for r in per_tet))
            delta_b = float(LB_global - UB_global)

        # 6) ms_a（沿用 ms 汇总，只作历史记录/可视化，不影响 active）
        if any(active_mask.values()):
            ms_a = float(min(r["ms"] for r in per_tet if active_mask[r["simplex_index"]]))
        else:
            ms_a = float(min(r["ms"] for r in per_tet))
        ms_iter = ms_a

        # === Print the current round's optimality gap ===
        gap_abs = float(UB_global - LB_global)
        gap_pct = (gap_abs / (abs(UB_global) + 1e-16)) * 100.0
        if verbose:
            print(f"[Iter {it}] Optimality gap: {gap_abs:.6e} ({gap_pct:.3f}%)")

        # 7) record
        LB_hist.append(LB_global)
        UB_hist.append(UB_global)
        ms_hist.append(ms_iter)
        node_count.append(len(nodes))
        UB_node_hist.append(UB_node)
        ms_a_hist.append(ms_a)
        ms_b_hist.append(delta_b)   # 新口径下把“ms_b”位置记录成 delta_b
        active_ratio_hist.append(active_ratio)

        # === Convergence stopping condition ===
        if gap_stop_tol is not None and float(gap_stop_tol) > 0.0:
            gap = float(UB_global - LB_global)
            if gap <= float(gap_stop_tol):
                if verbose:
                    print(f"[Iter {it}] Stop: UB-LB gap {gap:.6e} <= tol {float(gap_stop_tol):.6e}.")
                break

        # 8) print
        simp_with_min = [r["simplex_index"] for r in per_tet if ub_idx in r["vert_idx"]]
        if verbose:
            print(f"[Iter {it}] Active simplex ratio = {active_ratio:.6f}")
            print(f"[Iter {it}] UB node {UB_node} is in simplices {sorted(simp_with_min)}")
            src = f"T{ms_b_simp}" if ms_b_simp is not None else "N/A"
            print(f"[Iter {it}] LB = {LB_global:.6f} = UB({UB_global:.6f}) + delta_b({delta_b:.3e}) from {src}")

        # 9) Candidate ranking（仍使用 ms-based 候选）
        active = [r for r in per_tet if active_mask[r["simplex_index"]]]
        ub_active = [r for r in active if ub_idx in r["vert_idx"]]
        pool_records = ub_active if len(ub_active) > 0 else active

        cand_items = []
        for rec in pool_records:
            sid = rec["simplex_index"]
            ms_list = rec.get("ms_per_scene", [])
            pts_list = rec.get("xms_per_scene", [None]*len(ms_list))
            for s in range(len(ms_list)):
                cand_items.append({
                    "simplex_index": sid,
                    "scene": s,
                    "cand_ms": ms_list[s],
                    "cand_pt": pts_list[s],
                    "_rec": rec
                })

        def score_item(ci):
            ms = ci["cand_ms"]
            pt = ci["cand_pt"]
            d  = (float('inf') if pt is None else min_dist_to_nodes(pt, nodes))
            return (ms, -d)

        candidates_sorted = sorted(cand_items, key=score_item)

        if verbose:
            top_msg = "N/A"
            if len(candidates_sorted) > 0:
                t0 = candidates_sorted[0]
                top_msg = f"T{int(t0['simplex_index'])}, scene={t0['scene']}, ms={float(t0['cand_ms']):.3e}"
            print(f"[Iter {it}] candidate rank #1: {top_msg}")
            _print_candidates_table(candidates_sorted, nodes, topN=10)
            print()

        # 10) Select new point + collision handling
        new_node = None
        chosen_ms = None
        chosen_cand = None
        stop_due_to_collision = False

        def handle_collision(cand_pt, ci, stage_note="active"):
            nonlocal stop_due_to_collision
            X = np.asarray(nodes, float)
            P = np.asarray(cand_pt, float)
            dists = np.linalg.norm(X - P, axis=1)
            j_star = int(np.argmin(dists))
            d_star = float(dists[j_star])
            orange_ids = [r["simplex_index"] for r in per_tet if j_star in r["vert_idx"]]
            debug_pack = {
                "reason": "candidate_too_close",
                "iter": it,
                "stage": stage_note,
                "min_dist": float(min_dist),
                "closest_node_index": j_star,
                "closest_node_point": tuple(map(float, nodes[j_star])),
                "closest_distance": d_star,
                "cand_simplex": int(ci["simplex_index"]),
                "cand_scene": int(ci["scene"]),
                "cand_point": tuple(map(float, cand_pt)),
                "cand_ms": float(ci["cand_ms"]),
                "UB_global": float(UB_global),
                "LB_global": float(LB_global),
                "active_ratio": float(active_ratio),
                "UB_node": tuple(map(float, UB_node)),
                "active_mask": {int(k): bool(v) for k, v in active_mask.items()},
                "nodes_snapshot": [tuple(map(float, nd)) for nd in nodes],
                "per_tet_snapshot": [
                    {
                        "simplex_index": int(r["simplex_index"]),
                        "vert_idx": list(map(int, r["vert_idx"])),
                        "verts": [tuple(map(float, x)) for x in r['verts']],
                        "ms": float(r["ms"]),
                        "ms_per_scene": [float(x) for x in r.get("ms_per_scene", [])],
                        "LB": float(r["LB"]),
                        "UB": float(r["UB"]),
                        "best_scene": int(r["best_scene"]),
                        "x_ms_best_scene": tuple(map(float, r["x_ms_best_scene"])) if r.get("x_ms_best_scene") is not None else None,
                        "volume": float(r["volume"]),
                    } for r in per_tet
                ],
                "highlight_simplices": list(map(int, orange_ids)),
            }
            global LAST_DEBUG
            LAST_DEBUG = debug_pack
            plot_iteration_plotly(
                it, nodes, tri, active_mask, UB_node, cand_pt, per_tet,
                highlight_simplices=orange_ids
            )
            if verbose:
                print(
                    f"[STOP] Candidate {tuple(map(float, cand_pt))} "
                    f"(scene {ci['scene']}) is too close to existing node #{j_star} at distance {d_star:.3e} "
                    f"(< {min_dist:g}). Highlighted simplices: {sorted(orange_ids)}"
                )
            stop_due_to_collision = True

        for rank, ci in enumerate(candidates_sorted, start=1):
            cand_pt = ci["cand_pt"]
            if cand_pt is None:
                continue
            if min_dist_to_nodes(cand_pt, nodes) >= min_dist:
                new_node   = cand_pt
                chosen_ms  = ci["cand_ms"]
                chosen_cand= ci
                if verbose:
                    print(
                        f"Chosen node {tuple(map(float, cand_pt))} "
                        f"with ms={chosen_ms:.3e} "
                        f"(simp T{ci['simplex_index']}, scene {ci['scene']}, rank #{rank})"
                    )
                    print(f"[Iter {it}] next node comes from simplex T{int(ci['simplex_index'])}, scene {int(ci['scene'])}")
                break
            else:
                if verbose:
                    print(
                        f"Skip candidate {tuple(map(float, cand_pt))} "
                        f"(simp T{ci['simplex_index']}, scene {ci['scene']}, rank #{rank}) "
                        f"because too close to existing nodes (< {min_dist:g})."
                    )
                handle_collision(cand_pt, ci, stage_note="active")
                break

        if (new_node is None) and (not stop_due_to_collision) and (len(active) > 0):
            if verbose:
                print("[fallback] All UB-neighborhood candidates too close; try all active simplices × scenes...")
            cand_items_all = []
            for rec in active:
                sid = rec["simplex_index"]
                ms_list = rec.get("ms_per_scene", [])
                pts_list = rec.get("xms_per_scene", [None]*len(ms_list))
                for s in range(len(ms_list)):
                    cand_items_all.append({
                        "simplex_index": sid,
                        "scene": s,
                        "cand_ms": ms_list[s],
                        "cand_pt": pts_list[s],
                        "_rec": rec
                    })
            for ci in sorted(cand_items_all, key=score_item):
                cand_pt = ci["cand_pt"]
                if cand_pt is None:
                    continue
                if min_dist_to_nodes(cand_pt, nodes) >= min_dist:
                    new_node   = cand_pt
                    chosen_ms  = ci["cand_ms"]
                    chosen_cand= ci
                    if verbose:
                        print(
                            f"Chosen node {tuple(map(float, cand_pt))} "
                            f"with ms={chosen_ms:.3e} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}) [fallback-active]"
                        )
                    break
                else:
                    if verbose:
                        print(
                            f"Skip (active) candidate {tuple(map(float, cand_pt))} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}) "
                            f"because too close to existing nodes (< {min_dist:g})."
                        )
                    handle_collision(cand_pt, ci, stage_note="fallback-active")
                    break

        if stop_due_to_collision:
            if verbose:
                print(f"[Iter {it}] Stop due to collision.")
            break

        if new_node is None:
            if verbose:
                print("New node too close for all candidates (or infeasible ms); stop.")
            break

        # == Strong validation: Avoid next_node equal to vertex ==
        tol_same = 1e-10
        def _same(a, b, tol=tol_same):
            a = np.asarray(a, float); b = np.asarray(b, float)
            return np.linalg.norm(a - b) <= tol

        if chosen_cand is not None and "_rec" in chosen_cand:
            rec = chosen_cand["_rec"]
            offending_vert = None
            for v in rec["verts"]:
                if _same(new_node, v):
                    offending_vert = tuple(map(float, v))
                    break
            if offending_vert is not None:
                from utils import LAST_DEBUG
                LAST_DEBUG = {
                    "reason": "next_node_equals_vertex",
                    "iter": it,
                    "new_node": tuple(map(float, new_node)),
                    "offending_vertex": offending_vert,
                    "tol_same": tol_same,
                    "candidate": {
                        "simplex_index": int(rec["simplex_index"]),
                        "scene": int(chosen_cand["scene"]),
                        "vert_idx": list(map(int, rec["vert_idx"])),
                        "verts": [tuple(map(float, x)) for x in rec["verts"]],
                        "ms": float(chosen_cand["cand_ms"]),
                        "ms_per_scene": [float(x) for x in rec["ms_per_scene"]],
                        "best_scene": int(rec["best_scene"]),
                        "x_ms_best_scene": tuple(map(float, rec["x_ms_best_scene"])) if rec.get("x_ms_best_scene") is not None else None,
                        "LB": float(rec["LB"]),
                        "UB": float(rec["UB"]),
                        "volume": float(rec["volume"]),
                    },
                    "UB_global": float(UB_global),
                    "LB_global": float(LB_global),
                    "active_ratio": float(active_ratio),
                    "UB_node": tuple(map(float, UB_node)),
                    "active_mask": {int(k): bool(v) for k, v in active_mask.items()},
                    "nodes_snapshot": [tuple(map(float, nd)) for nd in nodes],
                    "per_tet_snapshot": [
                        {
                            "simplex_index": int(r["simplex_index"]),
                            "vert_idx": list(map(int, r["vert_idx"])),
                            "verts": [tuple(map(float, x)) for x in r["verts"]]
                            ,
                            "ms": float(r["ms"]),
                            "ms_per_scene": [float(x) for x in r.get("ms_per_scene", [])],
                            "LB": float(r["LB"]),
                            "UB": float(r["UB"]),
                            "best_scene": int(r["best_scene"]),
                            "x_ms_best_scene": tuple(map(float, r["x_ms_best_scene"])) if r.get("x_ms_best_scene") is not None else None,
                            "volume": float(r["volume"]),
                        } for r in per_tet
                    ],
                }
                # Highlight the simplex adjacent to the offending vertex.
                vert_idx_list = []
                for j, nd in enumerate(nodes):
                    if _same(offending_vert, nd):
                        vert_idx_list.append(j)
                orange_ids = [r["simplex_index"] for r in per_tet if any(j in r["vert_idx"] for j in vert_idx_list)]
                plot_iteration_plotly(it, nodes, tri, active_mask, UB_node, new_node, per_tet,
                                      highlight_simplices=orange_ids)
                if verbose:
                    print("[STOP] new_node coincides with a simplex vertex. Highlighted simplices:",
                          sorted(orange_ids))
                break

        # Visualization
        plot_iteration_plotly(it, nodes, tri, active_mask, UB_node, new_node, per_tet,
                              highlight_simplices=None)

        # add node and evaluate
        new_vals = []
        for ω in range(S):
            val = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], new_node)
            new_vals.append(val)

        nodes.append(tuple(map(float, new_node)))
        for ω in range(S):
            scen_values[ω].append(new_vals[ω])

        add_node_hist.append(new_node)
        if verbose:
            print(f"[Iter {it}] Elapsed: {perf_counter() - t_iter0:.3f}s")
        it += 1

    return {
        "nodes": np.array(nodes, float),
        "LB_hist": LB_hist,
        "UB_hist": UB_hist,
        "ms_hist": ms_hist,
        "ms_a_hist": ms_a_hist,
        "ms_b_hist": ms_b_hist,  # 现在存 delta_b
        "node_count": node_count,
        "UB_node_hist": UB_node_hist,
        "added_nodes": add_node_hist,
        "active_ratio_hist": active_ratio_hist,
    }
