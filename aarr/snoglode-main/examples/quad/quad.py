import pyomo.environ as pyo
import snoglode as sno
import os

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

scenarios = [
    'scenario1',
    'scenario2',
    'scenario3'
]

name_value_map = {
    'scenario1': 0,
    'scenario2': 1,
    'scenario3': 2
    }

name_prob_map = {
    'scenario1': 0.1,
    'scenario2': 0.1,
    'scenario3': 0.8
    }

def build_scenario_model(scenario_name):
    m  = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3,3))
    m.y = pyo.Var(bounds=(-100,100))
    m.obj = pyo.Objective(expr=(m.y-name_value_map[scenario_name])**2)
    m.c = pyo.Constraint(expr= m.x == m.y)

    return [m, 
            {'x': m.x}, 
            name_prob_map[scenario_name]]

def solve_extensive_form():
    m = pyo.ConcreteModel()

    scenario_data = dict()
    @m.Block(scenarios)
    def scenario_blocks(b, s):
        scenario_data[s] = build_scenario_model(s)
        return scenario_data[s][0]

    # get a list of all the "first-stage" variables
    first_stage_vars = set()
    for s in scenarios:
        first_stage_vars.update([k for k in scenario_data[s][1].keys()])

    for n in first_stage_vars:
        m.add_component(n, pyo.Var())

    @m.Constraint(scenarios, first_stage_vars)
    def nonanticpativity(m, s, n):
        if hasattr(m.scenario_blocks[s], n):
            return getattr(m.scenario_blocks[s],n) == getattr(m, n)

    @m.Objective()
    def objective(m):
        exp = 0
        for s in scenarios:
            exp += scenario_data[s][2]*m.scenario_blocks[s].obj
            m.scenario_blocks[s].obj.deactivate()
        return exp
    
    status = pyo.SolverFactory('gurobi').solve(m, tee=True)
    m.pprint()
    pyo.assert_optimal_termination(status)

def solve_decomposition():
    params = sno.SolverParameters(subproblem_names = scenarios,
                                  subproblem_creator = build_scenario_model,
                                  lb_solver = pyo.SolverFactory("gurobi"),
                                  cg_solver = pyo.SolverFactory("gurobi"),
                                  ub_solver = pyo.SolverFactory("gurobi"))
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm)
    params.activate_verbose()
    if (size==1): params.set_logging(fname = os.getcwd() + "/logs/quad_log")
    else: params.set_logging(fname = os.getcwd() + "/logs/quad_log_parallel")
    if (rank==0): params.display()
    
    solver = sno.Solver(params)
    solver.solve(max_iter=500)

    if (rank==0):
        print("\n====================================================================")
        print("SOLUTION")
        for n in solver.subproblems.names:
            print(f"subproblem = {n}")
            for vn in solver.solution.subproblem_solutions[n]:
                var_val = solver.solution.subproblem_solutions[n][vn]
                print(f"  var name = {vn}, value = {var_val}")
            print()
        print("====================================================================")

if __name__ == '__main__':
    solve_extensive_form()
    solve_decomposition()