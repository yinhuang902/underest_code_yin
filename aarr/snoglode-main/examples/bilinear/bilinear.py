"""
2-Scenario Bilinear Stochastic Program 
"""
import pyomo.environ as pyo
import numpy as np
import snoglode as sno
import os

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

num_scenarios = 5

class NonconvexOpt():
    def __init__(self, b1=4, y_ub=6, y_lb=0, x_ub=4, x_lb=0) -> None:
        self.b1 = b1
        self.y_ub = y_ub
        self.y_lb = y_lb
        self.x_ub = x_ub
        self.x_lb = x_lb
        self.build_bilinear_model()

    def build_bilinear_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Reals,
                      bounds=(self.x_lb,self.x_ub))      
        m.y = pyo.Var(within=pyo.Reals,
                      bounds=(self.y_lb,self.y_ub))
        m.obj = pyo.Objective( expr = ( - m.x - m.y ) )
        @m.Constraint()
        def c1_rule(m):
            return (m.x * m.y <= self.b1)
        self.m=m

class BilinearProblem():
    def __init__(self, c, y_ub, y_lb, x_ub, x_lb) -> None:
        assert y_lb <= y_ub
        assert x_lb <= x_ub
        self.c = c
        self.y_ub = y_ub
        self.y_lb = y_lb
        self.x_ub = x_ub
        self.x_lb = x_lb
        self.build_bilinear_model()

    def build_bilinear_model(self) -> None:
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Reals,
                      bounds=(self.x_lb,self.x_ub))      
        m.y = pyo.Var(within=pyo.Reals,
                      bounds=(self.y_lb,self.y_ub))
        m.obj = pyo.Objective( expr = ( - m.x - m.y ) )
        @m.Constraint()
        def c1_rule(m):
            return (m.x * m.y <= self.c)
        self.m=m


def subproblem_creator(subproblem_name):
    if subproblem_name=="model1":
        optmodel = NonconvexOpt(b1=4,
                                y_lb=0,
                                y_ub=4,
                                x_lb=0,
                                x_ub=6)
        model = optmodel.m
    if subproblem_name=="model2":
        optmodel = NonconvexOpt(b1=8,
                                y_lb=0,
                                y_ub=6,
                                x_lb=0,
                                x_ub=4)
        model = optmodel.m
    
    lifted_variable_ids = {"x": model.x}
    subproblem_probability = 1/2

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            subproblem_probability]             # probability of this subproblem


if __name__=="__main__":

    # organize subproblems
    subproblem_names = ["model1", "model2"]
    
    # solvers
    lb_solver = pyo.SolverFactory("gurobi")
    lb_solver.options["NonConvex"] = 2
    cg_solver = pyo.SolverFactory("gurobi")
    cg_solver.options["NonConvex"] = 2
    ub_solver = pyo.SolverFactory("gurobi")
    ub_solver.options["NonConvex"] = 2

    # initialize solver parameters
    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                  subproblem_creator = subproblem_creator,
                                  lb_solver = lb_solver,
                                  cg_solver = cg_solver,
                                  ub_solver = ub_solver)
    params.guarantee_global_convergence()
    params.set_bounds_tightening(fbbt=True, obbt=True)
    if (size==1): params.set_logging(fname = os.getcwd() + "/logs/bilinear_log")
    else: params.set_logging(fname = os.getcwd() + "/logs/bilinear_log_parallel")
    params.display()

    # init and solve
    solver = sno.Solver(params)
    solver.solve(max_iter=100,
                 collect_plot_info=False)
    
    print("\n====================================================================")
    print("SOLUTION")
    print(f"Obj: {solver.tree.metrics.ub}")
    for name in solver.subproblems.names:
        print(f"subproblem = {name}")
        for var_name in solver.solution.subproblem_solutions[name]:
            var_val = solver.solution.subproblem_solutions[name][var_name]
            print(f"  var name = {var_name}, value = {round(var_val, 5)}")
        print()
    print("====================================================================")