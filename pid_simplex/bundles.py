# bundles.py
import numpy as np
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.opt import SolverStatus, TerminationCondition

class BaseBundle:
    """每个场景的基础模型（计算真实Q）+ 持久化求解器"""
    def __init__(self, model: pyo.ConcreteModel, options: dict | None = None):
        self.model = model
        self.gp = GurobiPersistent()
        self.gp.set_instance(model)
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
        self.gp.set_objective(model.obj)
        if options:
            self.gp.set_gurobi_param('MIPGap', options.get('MIPGap', 1e-1))
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            self.gp.set_gurobi_param('NonConvex', options.get('NonConvex', 2))
            if 'TimeLimit' in options:
                self.gp.set_gurobi_param('TimeLimit', options['TimeLimit'])

    def eval_at(self, first_vars, first_vals):
        for v, val in zip(first_vars, first_vals):
            v.fix(float(val))
            self.gp.update_var(v)
        self.gp.solve(load_solutions=True)
        val = float(pyo.value(self.model.obj_expr))
        for v in first_vars:
            v.unfix()
            self.gp.update_var(v)
        return val

class MSBundle:
    """单场景 ms 子问题（持久化），对一个四面体求解。
    采用固定结构约束 + set_linear_coefficients，就地更新系数，避免删/建约束的高开销。
    """
    def __init__(self, model_base: pyo.ConcreteModel, first_vars, options: dict | None = None):
        m = model_base.clone()

        # ---- barycentric weights ----
        m.lam_index = pyo.RangeSet(0, 3)
        m.lam = pyo.Var(m.lam_index, domain=pyo.NonNegativeReals)
        m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in m.lam_index) == 1.0)

        # ---- locate first-stage vars in clone ----
        self.Kp = m.find_component(first_vars[0].name)
        self.Ki = m.find_component(first_vars[1].name)
        self.Kd = m.find_component(first_vars[2].name)
        if any(v is None for v in (self.Kp, self.Ki, self.Kd)):
            raise RuntimeError("克隆模型中找不到 Kp/Ki/Kd")

        # ---- mirrors (mutable Params) for logging/可视化（非必须，但便于调试/导出）----
        m.vx = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vy = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vz = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.fv = pyo.Param(m.lam_index, mutable=True, initialize=0.0)

        # ---- 固定结构的“链接约束”，先用 0 系数建模；之后用 set_linear_coefficients 更新 ----
        # 形式： Kp - sum(alpha_j * lam[j]) == 0  （初始 alpha_j=0）
        m.link_kp = pyo.Constraint(expr=self.Kp - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_ki = pyo.Constraint(expr=self.Ki - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_kd = pyo.Constraint(expr=self.Kd - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        # As: 聚合的“顶点函数值的凸组合”
        m.As = pyo.Var()
        m.As_def = pyo.Constraint(expr=m.As - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        # 目标：min (原场景目标 - As)
        if hasattr(m, 'obj'):
            m.del_component('obj')
        m.obj = pyo.Objective(expr=m.obj_expr - m.As, sense=pyo.minimize)

        # ---- persistent solver ----
        self.model = m
        self.gp = GurobiPersistent()
        self.gp.set_instance(m)
        self.gp.set_objective(m.obj)
        if options:
            self.gp.set_gurobi_param('MIPGap', options.get('MIPGap', 1e-1))
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            self.gp.set_gurobi_param('NonConvex', options.get('NonConvex', 2))
            if 'TimeLimit' in options:
                self.gp.set_gurobi_param('TimeLimit', options['TimeLimit'])

        # 方便访问
        self.lam = m.lam
        self.link_kp = m.link_kp
        self.link_ki = m.link_ki
        self.link_kd = m.link_kd
        self.As     = m.As
        self.As_def = m.As_def
        self._V_cached = None  # [(x,y,z)]*4

    # ---- 内部：一次性更新某个“链接约束”的 lam 系数（在 LHS） ----
    def _set_link_coeffs(self, con, coeffs):
        """把约束 con 中 lam[j] 的系数改成 coeffs[j]（LHS 系数）。
        通过 Gurobi 模型的 chgCoeff 直接修改矩阵，兼容没有 set_linear_* 的 Pyomo。"""
        # 取到底层 Gurobi 模型与 Pyomo→Gurobi 的映射
        gmodel = getattr(self.gp, "_solver_model", None)
        con_map = getattr(self.gp, "_pyomo_con_to_solver_con_map", None)
        var_map = getattr(self.gp, "_pyomo_var_to_solver_var_map", None)
        if gmodel is None or con_map is None or var_map is None:
            # 老环境实在没有映射，就退回兜底方案（下面给出）
            raise AttributeError("找不到持久化求解器的内部映射，无法用 chgCoeff。")

        grb_con = con_map[con]
        # 先清零原系数（保险）
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, 0.0)
        # 再写新系数
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, float(coeffs[j]))
        gmodel.update()

    def update_tetra(self, tet_vertices, fverts_scene):
        V = [tuple(map(float, tet_vertices[j])) for j in range(4)]
        F = [float(fverts_scene[j]) for j in range(4)]
        self._V_cached = V

        vx = [V[j][0] for j in range(4)]
        vy = [V[j][1] for j in range(4)]
        vz = [V[j][2] for j in range(4)]

        # 可选：Param 仅用于记录
        for j in range(4):
            self.model.vx[j] = vx[j]
            self.model.vy[j] = vy[j]
            self.model.vz[j] = vz[j]
            self.model.fv[j] = F[j]

        # LHS：K? - Σ(a_j * lam[j]) == 0  ⇒ lam 的系数是 -a_j
        self._set_link_coeffs(self.link_kp, [-x for x in vx])
        self._set_link_coeffs(self.link_ki, [-y for y in vy])
        self._set_link_coeffs(self.link_kd, [-z for z in vz])
        self._set_link_coeffs(self.As_def,  [-f for f in F])

    def solve(self):
        res = self.gp.solve(load_solutions=True)
        ok = (res.solver.status == SolverStatus.ok) and \
             (res.solver.termination_condition in {
                 TerminationCondition.optimal,
                 TerminationCondition.locallyOptimal
             })
        return ok

    def get_ms_and_point(self):
        ms_val = float(pyo.value(self.model.obj))
        lam_star = np.array([pyo.value(self.lam[j]) for j in range(4)], dtype=float)
        V = np.array(self._V_cached, dtype=float)
        new_pt = lam_star @ V
        return ms_val, lam_star, tuple(map(float, new_pt))
