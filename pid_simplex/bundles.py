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

    This wrapper:
      1) Installs a standard Pyomo Objective component from `model.obj_expr`
         (replacing any pre-existing `model.obj`), and
      2) Binds the model to `GurobiPersistent` so variables/constraints/objective
         updates do not rebuild the entire model each time.

    Parameters
    ----------
    model : pyo.ConcreteModel
        A fully specified Pyomo model that exposes an expression `model.obj_expr`
        to be minimized. If an `obj` component already exists it will be removed
        and replaced to ensure consistency with the persistent solver.
    options : dict | None, optional
        Gurobi parameters to set on the persistent solver. Recognized keys:
        - 'MIPGap' (float, default 1e-1)
        - 'NumericFocus' (int {0..3}, default 1)
        - 'Presolve' (int, default 2)
        - 'NonConvex' (int, default 2)
        - 'TimeLimit' (float, seconds)

    Attributes
    ----------
    model : pyo.ConcreteModel
        The bound (and possibly lightly modified) Pyomo model.
    gp : GurobiPersistent
        The persistent solver instance bound to `model`.

    Methods
    -------
    eval_at(first_vars, first_vals) -> float
        Fixes the provided first-stage variables to `first_vals`, solves the
        model, reads the objective value at `model.obj_expr`, and then unfixes
        the variables. Returns the scalar objective value.

    Notes
    -----
    - `eval_at` uses `gp.update_var(v)` after fixing/unfixing to keep the
      persistent model in sync without triggering a full rebuild.
    - This class does not alter constraints or variable domains; it only
      installs the objective and manages re-solves efficiently.
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
    subproblem using a fixed-structure formulation with barycentric weights.
    Coefficients in the linking constraints are updated in-place (matrix edits)
    to avoid rebuild overhead.

    Problem sketch
    --------------
    - Introduce barycentric weights `lam[j]` (j=0..3), sum to 1, lam[j] >= 0.
    - Link first-stage variables (Kp, Ki, Kd) to the convex combination of the
      4 vertices of a tetrahedron via linear constraints:
          Kp - sum(x_j * lam[j]) = 0,
          Ki - sum(y_j * lam[j]) = 0,
          Kd - sum(z_j * lam[j]) = 0.
    - Define `As` as the convex combination of per-vertex function values:
          As - sum(f_j * lam[j]) = 0.
    - Objective:
          minimize  model.obj_expr - As
      which encourages selecting a point whose scenario objective is small
      relative to the convex combination of vertex values.

    Parameters
    ----------
    model_base : pyo.ConcreteModel
        A base model that exposes:
          - first-stage variables present by name (e.g., Kp, Ki, Kd),
          - an expression `obj_expr` to minimize.
        This model is cloned internally so the MS subproblem can alter
        constraints/objective without touching the original.
    first_vars : Sequence[pyo.Var]
        A 3-tuple/list of the first-stage variables (Kp, Ki, Kd) from
        `model_base`. Their **names** are used to locate the counterparts in
        the cloned model (via `find_component`).
    options : dict | None, optional
        Gurobi parameters for the persistent solver. Recognized keys:
        'MIPGap', 'NumericFocus', 'Presolve', 'NonConvex', 'TimeLimit'.

    Attributes
    ----------
    model : pyo.ConcreteModel
        The cloned and augmented model (barycentric vars/constraints installed).
    gp : GurobiPersistent
        The persistent solver instance bound to `model`.
    lam : pyo.Var
        Barycentric weights (index 0..3), enforce convex combination.
    link_kp, link_ki, link_kd : pyo.Constraint
        Linking constraints tying (Kp,Ki,Kd) to the tetra vertices.
    As : pyo.Var
        Convex combination of vertex function values.
    As_def : pyo.Constraint
        Definition constraint for `As`.
    _V_cached : list[tuple[float, float, float]] | None
        Cached coordinates of the current tetrahedron vertices, used to map
        the optimal barycentric weights back to a Cartesian point.

    Methods
    -------
    update_tetra(tet_vertices, fverts_scene) -> None
        Update the in-place coefficients of the linking constraints to reflect
        the current tetrahedron geometry and vertex function values for THIS
        SCENE. This avoids rebuilding constraints.
    solve() -> bool
        Solve the current subproblem; returns True if the solve ended with an
        optimal or locally optimal termination condition.
    get_ms_and_point() -> tuple[float, np.ndarray, tuple[float, float, float]]
        Read the scalar ms value (objective), the optimal barycentric weights,
        and the corresponding Cartesian point (Kp,Ki,Kd).

    Notes
    -----
    - Coefficients are updated directly on the underlying Gurobi model using
      `chgCoeff` via the persistent solver's internal maps. This is more robust
      across Pyomo versions that may not expose `set_linear_coefficients`.
    - The class is intentionally single-scenario: instantiate one `MSBundle`
      per scenario, then feed scenario-specific vertex values to `update_tetra`.
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

        # ---- mirrors (mutable Params) for logging----
        m.vx = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vy = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vz = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.fv = pyo.Param(m.lam_index, mutable=True, initialize=0.0)

        # ---- fixed structure "link constraints", firstly model them with 0 coefficients; then update them using set_linear_coefficients. ----
        # form： Kp - sum(alpha_j * lam[j]) == 0  （initial alpha_j=0）
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

    # ---- Update the LAM coefficient of the "link constraint" in a single operation (in LHS) ----
    def _set_link_coeffs(self, con, coeffs):
        """
        Internal: set the coefficients of `lam[j]` on the LHS of a given linear
        constraint to `coeffs[j]` via direct edits to the Gurobi matrix.

        This path uses the persistent solver's internal mapping
        (`_pyomo_con_to_solver_con_map`, `_pyomo_var_to_solver_var_map`) and
        `chgCoeff` on the underlying Gurobi model to avoid relying on Pyomo
        version-specific helper APIs.

        Parameters
        ----------
        con : pyo.Constraint
            The linear constraint whose LHS lam-coefficients will be updated.
        coeffs : Sequence[float]
            Length-4 sequence specifying the new coefficients for lam[0..3].

        Raises
        ------
        AttributeError
            If the persistent solver does not expose the internal maps needed
            to access the underlying Gurobi objects.
        """
        # Obtain the mapping between the underlying Gurobi model and Pyomo→Gurobi.
        gmodel = getattr(self.gp, "_solver_model", None)
        con_map = getattr(self.gp, "_pyomo_con_to_solver_con_map", None)
        var_map = getattr(self.gp, "_pyomo_var_to_solver_var_map", None)
        if gmodel is None or con_map is None or var_map is None:
            raise AttributeError("The internal mapping of the persistent solver could not be found, and chgCoeff cannot be used.")

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

        # Param only for record
        for j in range(4):
            self.model.vx[j] = vx[j]
            self.model.vy[j] = vy[j]
            self.model.vz[j] = vz[j]
            self.model.fv[j] = F[j]

        # LHS：K? - Σ(a_j * lam[j]) == 0  ⇒ lam coefficient is -a_j
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
