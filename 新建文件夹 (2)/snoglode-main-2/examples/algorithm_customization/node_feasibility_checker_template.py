"""
The user can specify a simple feasibility check on a given node
of the spatial Branch and Bound tree.

The intuition here is to avoid setting an entire state of models
and having to pass the new model to a solver to determine infeasibility
if there are simple, cheap tests that the user can exploit based on 
their specific problem structure. 

The function takes in the current node of the spatial branch and bound tree,
which defines what bounds there are on the branching variables. 
It must return a boolean (True if feasible; False if infeasible).

NOTE: this is a very critical implementation; if it is done
      incorrectly, i.e., things are marked infeasible that are in
      fact feasibile, you will destory the sBnB guarantees.
"""
import snoglode as sno

def node_feasibility_check(node: sno.Node) -> bool:

    raise NotImplementedError

    if (some_condition): return False
    else: return True