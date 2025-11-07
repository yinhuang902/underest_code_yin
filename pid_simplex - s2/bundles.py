# bundles.py
import numpy as np
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.opt import SolverStatus, TerminationCondition


class BaseBundle:
    """
    Base bundle for a single scenario: holds a Pyomo model that evaluates the
    true objective (Q) at given first-stage values, backed by a persistent
    Gurobi solver for fast resolves.
    """
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
    """
    Single-scenario MS subproblem (persistent): solves the tetrahedral
    subproblem using barycentric weights lam[0..3] with objective min (Q - As).
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
            raise RuntimeError("Can't find Kp/Ki/Kd in clone model")

        # ---- mirrors (mutable Params) for logging ----
        m.vx = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vy = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vz = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.fv = pyo.Param(m.lam_index, mutable=True, initialize=0.0)

        # ---- link constraints (coeffs updated in-place) ----
        m.link_kp = pyo.Constraint(expr=self.Kp - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_ki = pyo.Constraint(expr=self.Ki - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_kd = pyo.Constraint(expr=self.Kd - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        m.As = pyo.Var()
        m.As_def = pyo.Constraint(expr=m.As - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        # obj：min (Qs - As)
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

        self.lam = m.lam
        self.link_kp = m.link_kp
        self.link_ki = m.link_ki
        self.link_kd = m.link_kd
        self.As     = m.As
        self.As_def = m.As_def
        self._V_cached = None  # [(x,y,z)]*4

    def _set_link_coeffs(self, con, coeffs):
        gmodel = getattr(self.gp, "_solver_model", None)
        con_map = getattr(self.gp, "_pyomo_con_to_solver_con_map", None)
        var_map = getattr(self.gp, "_pyomo_var_to_solver_var_map", None)
        if gmodel is None or con_map is None or var_map is None:
            raise AttributeError("The internal mapping of the persistent solver could not be found.")
        grb_con = con_map[con]
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, 0.0)
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

        # record
        for j in range(4):
            self.model.vx[j] = vx[j]
            self.model.vy[j] = vy[j]
            self.model.vz[j] = vz[j]
            self.model.fv[j] = F[j]

        # K? - Σ(a_j * lam[j]) == 0  ⇒ lam coeff = -a_j
        self._set_link_coeffs(self.link_kp, [-x for x in vx])
        self._set_link_coeffs(self.link_ki, [-y for y in vy])
        self._set_link_coeffs(self.link_kd, [-z for z in vz])
        # As - Σ(f_j * lam[j]) == 0  ⇒ lam coeff = -f_j
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
        lam_star = np.array([pyo.value(self.model.lam[j]) for j in range(4)], dtype=float)
        V = np.array(self._V_cached, dtype=float)
        new_pt = lam_star @ V
        return ms_val, lam_star, tuple(map(float, new_pt))


class QMinBundle:
    """
    True-Q minimization on a tetrahedron: same lam[0..3] linking constraints,
    but objective is the raw model.obj_expr (no As term).
    """
    def __init__(self, model_base: pyo.ConcreteModel, first_vars, options: dict | None = None):
        m = model_base.clone()

        m.lam_index = pyo.RangeSet(0, 3)
        m.lam = pyo.Var(m.lam_index, domain=pyo.NonNegativeReals)
        m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in m.lam_index) == 1.0)

        self.Kp = m.find_component(first_vars[0].name)
        self.Ki = m.find_component(first_vars[1].name)
        self.Kd = m.find_component(first_vars[2].name)
        if any(v is None for v in (self.Kp, self.Ki, self.Kd)):
            raise RuntimeError("Can't find Kp/Ki/Kd in clone model")

        m.vx = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vy = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vz = pyo.Param(m.lam_index, mutable=True, initialize=0.0)

        m.link_kp = pyo.Constraint(expr=self.Kp - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_ki = pyo.Constraint(expr=self.Ki - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_kd = pyo.Constraint(expr=self.Kd - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        if hasattr(m, 'obj'):
            m.del_component('obj')
        m.obj = pyo.Objective(expr=m.obj_expr, sense=pyo.minimize)

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

        self.lam = m.lam
        self.link_kp = m.link_kp
        self.link_ki = m.link_ki
        self.link_kd = m.link_kd
        self._V_cached = None

    def _set_link_coeffs(self, con, coeffs):
        gmodel = getattr(self.gp, "_solver_model", None)
        con_map = getattr(self.gp, "_pyomo_con_to_solver_con_map", None)
        var_map = getattr(self.gp, "_pyomo_var_to_solver_var_map", None)
        if gmodel is None or con_map is None or var_map is None:
            raise AttributeError("The internal mapping of the persistent solver could not be found.")
        grb_con = con_map[con]
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, 0.0)
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, float(coeffs[j]))
        gmodel.update()

    def update_tetra(self, tet_vertices):
        V = [tuple(map(float, tet_vertices[j])) for j in range(4)]
        self._V_cached = V
        vx = [V[j][0] for j in range(4)]
        vy = [V[j][1] for j in range(4)]
        vz = [V[j][2] for j in range(4)]

        for j in range(4):
            self.model.vx[j] = vx[j]
            self.model.vy[j] = vy[j]
            self.model.vz[j] = vz[j]

        self._set_link_coeffs(self.link_kp, [-x for x in vx])
        self._set_link_coeffs(self.link_ki, [-y for y in vy])
        self._set_link_coeffs(self.link_kd, [-z for z in vz])

    def solve(self):
        res = self.gp.solve(load_solutions=True)
        ok = (res.solver.status == SolverStatus.ok) and \
             (res.solver.termination_condition in {
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal
             })
        return ok

    def get_qmin_and_point(self):
        q_val = float(pyo.value(self.model.obj))
        lam_star = np.array([pyo.value(self.model.lam[j]) for j in range(4)], dtype=float)
        V = np.array(self._V_cached, dtype=float)
        new_pt = lam_star @ V
        return q_val, lam_star, tuple(map(float, new_pt))


# ---- convenience wrappers ----

def ms_on_tetra_for_scene(ms_bundle: MSBundle, tet_vertices, fverts_scene):
    ms_bundle.update_tetra(tet_vertices, fverts_scene)
    ok = ms_bundle.solve()
    if not ok:
        return float('inf'), None, None
    ms_val, lam_star, new_pt = ms_bundle.get_ms_and_point()
    return ms_val, lam_star, new_pt


def qmin_on_tetra_for_scene(qmin_bundle: QMinBundle, tet_vertices):
    qmin_bundle.update_tetra(tet_vertices)
    ok = qmin_bundle.solve()
    if not ok:
        return float('inf'), None, None
    q_val, lam_star, new_pt = qmin_bundle.get_qmin_and_point()
    return q_val, lam_star, new_pt

