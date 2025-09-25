"""
testing branch and bound - binaries
"""
import pyomo.environ as pyo
import snoglode as sno
import os

def ip_pyomo_model():
    """
    Builds basic Integer program
    See: Ignacaio's textbook, pg. 78 for the model and depiction of the 
         branch and bound tree
    """
    m = pyo.ConcreteModel()
    i = [1,2,3]
    m.y = pyo.Var(i,
                  within=pyo.Binary)

    m.obj=pyo.Objective( expr = (m.y[1] + 2*m.y[2] + 4*m.y[3]), 
                         sense=pyo.minimize )
    
    def c1_rule(m):
        return ( sum(m.y[ind] for ind in i) >= 1 )
    m.c1=pyo.Constraint( expr = c1_rule )

    def c2_rule(m):
        return ( m.y[1] - m.y[2] - m.y[3] <= 0 )
    m.c2=pyo.Constraint( expr = c2_rule )

    def c3_rule(m):
        return ( m.y[1] >= m.y[2] + 0.2 )
    m.c3=pyo.Constraint( expr = c3_rule )
    return m

def subproblem_creator(subproblem_name):
    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the list of first stage variables
        3) probability
    and returns as a list in this order.
    """
    # create parameters / model stored in obj for this scenario
    model = ip_pyomo_model()

    # grab the list of first stage variables
    lifted_variable_ids = {("y", i): model.y[i] \
                                for i in [1,2,3]}

    # probability of this particular scenario occuring
    scenario_probability = 1.0

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


if __name__=="__main__":
    
    subproblem_names = ["determistic"]
    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                subproblem_creator = subproblem_creator,
                                lb_solver = pyo.SolverFactory("gurobi"),
                                cg_solver = pyo.SolverFactory("gurobi"),
                                ub_solver = pyo.SolverFactory("gurobi"))
    params.set_bounds_tightening(fbbt = False)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.relax_binaries()
    params.activate_verbose()
    params.set_logging(fname = os.getcwd() + "/logs/ip_log")
    params.display()

    solver = sno.Solver(params)
    solver.solve()

    print("\n====================================================================")
    print("SOLUTION")
    for name in solver.subproblems.names:
        print(f"subproblem = {name}")
        for var_name in solver.solution.subproblem_solutions[name]:
            var_val = solver.solution.subproblem_solutions[name][var_name]
            print(f"  var name = {var_name}, value = {var_val}")
        print()
    print("====================================================================")