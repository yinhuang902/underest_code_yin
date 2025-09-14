import pyomo.environ as pyo
import numpy as np
import snoglode as sno
import os
import matplotlib.pyplot as plt

from pmedian_problem import PMedian

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def pmedian_subproblem_creator(subproblem_name):
    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the list of first stage variables
        3) probability
    and returns as a list in this order.
    """
    # unpack
    _,nb_facilities,_,max_facilities,_,total_communities,_,nb_subproblems,_,subproblem_nb = subproblem_name.split("_")
    
    # convert from str -> proper dtype
    nb_facilities = int(nb_facilities)
    max_facilities = int(max_facilities)
    total_communities = int(total_communities)
    nb_subproblems = int(nb_subproblems)

    # create parameters / model stored in obj for this scenario
    pmedian = PMedian(nb_facilities=nb_facilities,
                      max_facilities=max_facilities,
                      total_communities=total_communities,
                      nb_subproblems=nb_subproblems)
    model = pmedian.pmedian_pyomo_model(subproblem_nb)

    # facility placements = first stage
    lifted_variable_ids = {("facility", i): model.x[i] \
                                for i in pmedian.facilities}

    # probability of this particular scenario occuring
    scenario_probability = 1.0

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


def node_feasibility_check(node: sno.Node) -> bool:
        # if we have more than max_facilities fixed to 1, then this node 
        # and all subsequent children must be infeasible as well.
        
        nb_facilities_fixed_to_1 = 0 
        for var in node.state[sno.SupportedVars.binary]:
            if node.state[sno.SupportedVars.binary][var].is_fixed \
                        and node.state[sno.SupportedVars.binary][var].value == 1:
                nb_facilities_fixed_to_1 += 1

        if (nb_facilities_fixed_to_1 > 1): 
            return False
        
        else: return True


if __name__=="__main__":
    nb_facilities = 4
    max_facilities = 1
    total_communities = 10
    nb_subproblems = 2

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]

    # set up solver
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=pmedian_subproblem_creator,
                                  cg_solver = pyo.SolverFactory("gurobi"))
    params.set_bounds_tightening(fbbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.add_node_feasibility_checker(node_feasibility_check)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_queue_strategy(sno.QueueStrategy.lifo)
    params.relax_binaries()
    if (rank==0): params.display()

    if (size==1): params.set_logging(fname = os.getcwd() + "/logs/pmedian_log")
    else: params.set_logging(fname = os.getcwd() + "/logs/pmedian_log_parallel")

    # solve with given tolerances
    solver = sno.Solver(params)
    solver.solve(max_iter=20, # 2^4 = 16 nodes possible
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=True)

    if (rank==0):
        num_iters = np.arange(solver.iteration)
        plt.plot(num_iters, solver.plotter.iter_lb, label = "lb", color="blue")
        plt.plot(num_iters, solver.plotter.iter_ub, label = "ub", color="red")
        plt.legend()
        plt.xlabel("SNoGlode iteration")
        plt.xticks(num_iters)
        plt.ylabel("objective value")
        plt.title("SNoGlode Gap per Iteration: Average CG")
        plt.savefig("output/pmedian_iteration.png")