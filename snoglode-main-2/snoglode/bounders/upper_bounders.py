"""
Different methods for generating candidate solutions.

There are any number of ways to go about this, here we have
solving the EF and averaging the current solution.

This is very problem specific - can also define a custom candidate
generator class to determine this as well.
"""
import math
from typing import Tuple

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

# suppress warnings when loading infeasible models
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR) 

from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
from snoglode.bounders.base import BoundingProblemBase
from snoglode.utils.solve_stats import OneUpperBoundSolve
from snoglode.utils.ef import ExtensiveForm
import snoglode.utils.compute as compute

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class UpperBounder(BoundingProblemBase):
    """
    Class for solving an upper bounding problem with a pre-defined
    candidate generator.

    NOTE: this class required a candidate generator, but solves
          for the subproblems in a standardized manner.
    """
    perform_fbbt = True

    def __init__(self, 
                 candidate_solution_finder, 
                 subproblems: Subproblems,
                 ub_solver, 
                 candidate_solution_solver,
                 ub_time: float = 600, 
                 candidate_generator_time: float = 600) -> None:
        """
        Initializes the solver information and the candidate generator.

        Parameters
        -----------
        candidate_solution_finder : <class> 
            A child of the AbstractCandidateGenerator class.
        subproblems : snoglode.components.Subproblems
            Initialized subproblems
        ub_solver : pyo.SolverFactory
            initialized Pyomo solver factory object 
            to be used for the UPPER bounding problem solves.
        candidate_solution_solver : pyo.SolverFactory
            initialized Pyomo solver factory object 
            to be used for the CANDIDATE GENERATION solves.
        ub_time : int
            max time intended for solving the UB problem.
        candidate_generator_time : int
            max time intended for solving the CG problem.
        """
        # see snoglode.bounders.base.BoundingProblemBase
        super().__init__(solver = ub_solver, 
                         time_ub = ub_time)
    
        # we require a candidate generator plugin - either custom made class or one of those offered
        self.candidate_solution_finder = candidate_solution_finder(subproblems = subproblems,
                                                                   solver = candidate_solution_solver,
                                                                   time_ub = candidate_generator_time)

    def solve(self, 
              node: Node, 
              subproblems: Subproblems, 
              candidate_solution: dict) -> bool:
        """
        The "upper bounding" problem

         (1) drop non-anticipativity constraints
         (2) adjust lifted variable bounds to reflect current node state
         (3) fix to the candidate solution
         (4) solve each subproblem independently

        Parameters
        -----------
        node : Node
            This the current Node; it describes the current state
            in the spacial BnB tree we are in. It describes the bounds
            and fixed states of all lifted variables.
        subproblems : Subproblems
            initialized subproblems.
        candidate_solution : dict            
            for each of the lifted variable names, an associated value to fix solution to.

        Returns
        -----------
        feasible : bool
            If the upper bound solve was feasible
        """
        assert type(node)==Node
        assert type(subproblems)==Subproblems

        # init statistics object for this solve
        statistics = OneUpperBoundSolve(subproblems.names)
        
        # for each subproblems's model
        for subproblem_name in subproblems.names:

            # un-relax binaries, if there are any
            if subproblems.relax_binaries: subproblems.unrelax_all_binaries()
            if subproblems.relax_integers: subproblems.unrelax_all_integers()
            
            # fix the first stage variables to the candidate solution
            self.fix_to_candidate_solution(subproblem_lifted_variables = subproblems.subproblem_lifted_vars[subproblem_name], 
                                           subproblem_var_map = subproblems.var_to_data,
                                           candidate_solution_state = candidate_solution)

            # solve the current model representing this scenario
            subproblem_is_feasible, subproblem_objective = \
                self.solve_subproblem(subproblem_model = subproblems.model[subproblem_name])

            # if we have one infeasible scenario, the entire node is infeasible
            if not subproblem_is_feasible:
                
                # if we are infeasible, both UB/LB are infeasible -> add appropriate stats
                node.ub_problem.is_infeasible()
                
                return False
            
            # if we are feasible, add statistics
            statistics.update(subproblem_name = subproblem_name,
                              subproblem_objective = subproblem_objective,
                              subproblem_probabilty = subproblems.probability[subproblem_name])
            
        # if we were successful, add statistics to node
        node.ub_problem.is_feasible(pyo.value(statistics.aggregated_objective))

        return True

    
    def fix_to_candidate_solution(self, 
                                  subproblem_lifted_variables: list, 
                                  subproblem_var_map: pyo.ComponentMap,
                                  candidate_solution_state: dict) -> None:
        """
        Given the candidate solution state dictionary, fix all of the scenarios
        to the necessary values. 

        Parameters
        -----------
        subproblem_lifted_variables : list
            list of all the first stage variables. 
        candidate_solution : dict
            Dictionary containing all candidate solution values.
        """
        # for each of the first stage variables, retrieve value & fix
        for var in subproblem_lifted_variables:
            _, var_id, _ = subproblem_var_map[var]
            var_candidate_value = candidate_solution_state[var_id]
            var.fix(var_candidate_value)

    
    def solve_subproblem(self, 
                         subproblem_model: pyo.ConcreteModel) -> Tuple[bool, float]:
        """
        Given a Pyomo model representing one of the subproblems, solve

        Parameters
        -----------
        subproblem_model : pyo.ConcreteModel()
            A pyomo model representing a single subproblem. 
            Should be reflecting current node state 
        
        Returns
        -----------
        feasible_solution : bool
            Was the model feasible / solved okay?
        scenario_objective : float
            The value of the objective; returns None if infeasible.
        """
        # solve model
        results = self.opt.solve(subproblem_model,
                                 load_solutions = False, 
                                 symbolic_solver_labels=True,
                                 tee = False)

        # if the solution is optimal, return objective value
        if results.solver.termination_condition==TerminationCondition.optimal and \
            results.solver.status==SolverStatus.ok:

            # load in solutions, return [feasibility = True, obj, results]
            subproblem_model.solutions.load_from(results)

            # there should only be one objective, so return that value.
            return True, pyo.value(get_active_objective(subproblem_model))
        
        # if the solution is not feasible, return None
        else:
            return False, None


class AbstractCandidateGenerator(BoundingProblemBase):
    """
    Abstract base class for the candidate generators.

    This is not intended to be used directly -> a child class must be
    defined from this abstract parent to be used to solve for a candidate solution
    within the broaded solver.
    """

    def __init__(self, 
                 solver, 
                 subproblems: Subproblems, 
                 time_ub: float) -> None:
        """
        Initializes the solver information.

        Parameters
        -----------
        solver : pyo.SolverFactory
            initialized Pyomo solver factory object 
            to be used for the CANDIDATE GENERATION solves.
        subproblems : snoglode.components.Subproblems
            Initialized subproblems
        time_ub : int
            max time intended for solving the CG problem.
        """

        # see snoglode.bounders.base.BoundingProblemBase
        super().__init__(solver = solver, 
                         time_ub = time_ub)
        
        # grab all lifted var IDs - need to return a dict with these values
        # from the generate() function
        self.lifted_var_ids = subproblems.lifted_var_ids
        
        # this indicates if we resolve using the UB problem format or not
        # NOTE: this will only be False in very specific situations (i.e., EF)
        self.ub_required = True


    def generate(self, 
                 node: Node, 
                 subproblems: Subproblems) -> Tuple[bool, dict, float]:
        """
        This acts as a wrapper around generate_candidate, which
        is either user specified or selected among the given methods.

        Parameters
        ----------
        node : Node
            node object representing the current node we are exploring in the branch 
            and bound tree. Contains all bounding information.
        subproblems : Subproblems
            initialized subproblem manager.
            contains all subproblem names, models, probabilities, and lifted var lists/

        Returns
        ----------
        candidate_found : bool
            If this method successfully found a candidate or not
        candidate_solution : dict
            For each of the lifted variable ID's, therre should be a 
            corresponding value to fix the variable to.
        candidate_solution_obj : float
            If we do not want a global guarantee, an objective value is necessary
            to be produced.
        """
        assert type(node)==Node
        assert type(subproblems)==Subproblems
        return self.generate_candidate(node, subproblems)
    
    
    def generate_candidate(self,
                           node: Node,
                           subproblems: Subproblems) -> Tuple[bool, dict, float]:
        """
        This must be defined in the child class.
        It must always take these inputs, to maintain fluidity within the solver.
        
        Parameters
        ----------
        node : Node
            node object representing the current node we are exploring in the branch 
            and bound tree. Contains all bounding information.
        subproblems : Subproblems
            initialized subproblem manager.
            contains all subproblem names, models, probabilities, and lifted var lists/

        Returns
        ----------
        candidate_found : bool
            If this method successfully found a candidate or not
        candidate_solution : dict
            For each of the lifted variable ID's, therre should be a 
            corresponding value to fix the variable to.
        candidate_solution_obj : float
            If we do not want a global guarantee, an objective value is necessary
            to be produced.
        """

        print( "The child CG class must have a method called " + \
            "generate_candidate(node: Node, subproblems: Subproblems) -> feasible: bool, obj: float")
        raise NotImplementedError
        return candidate_found, candidate_solution, candidate_solution_obj

# ================================================================================================ #

class AverageLowerBoundSolution(AbstractCandidateGenerator):
    """
    Class for solving for taking the solution at each of the LB subproblems, 
    averaging, and fixing to see if we have a solution.
    """

    def __init__(self, 
                 solver, 
                 subproblems: Subproblems, 
                 time_ub: int) -> None:
        super().__init__(solver = solver, 
                         subproblems = subproblems, 
                         time_ub = time_ub)
        self.ub_required = True
        

    def generate_candidate(self, 
                           node: Node, 
                           subproblems: Subproblems) -> Tuple[bool, dict, float]:
        """
        Given a node, collect the LB solutions and average.
            see: average_lowerbound_solution() above.
        """
        return [True, 
                compute.average_lb_solution(node, subproblems), 
                math.nan]
    
# ================================================================================================ #

class SolveExtensiveForm(AbstractCandidateGenerator):
    """
    Class for solving for trying to solve the EF to local optimality.
    If you can solve it to global optimality already, why are you doing this...
    """

    def __init__(self, 
                 solver, 
                 subproblems: Subproblems, 
                 time_ub: float) -> None:
        super().__init__(solver = solver, 
                         subproblems = subproblems, 
                         time_ub = time_ub)
        self.ub_required = False

        # build the EF on the subproblem manager
        assert isinstance(subproblems, Subproblems)
        if not hasattr(subproblems, "ef"):
            if subproblems.verbose: print("Using EF method for candidate solution. No EF bound. Building EF.")
            subproblems.ef = ExtensiveForm(subproblems)

        # if this is parallel, track best objs so we explore more
        if (size > 1):
            self.best_objs = []
        

    def generate_candidate(self, 
                           _, 
                           subproblems: Subproblems) -> Tuple[bool, dict, float]:
        """
        Given a node, generate the local EF and solve for this
        particular region.

        Method: 
            (1) sets EF state -> do not need to actually do this step
                                 bc should already be set from LB solve.
            (2) activates EF
            (3) solves EF
            (4) deactives
        """
        
        # (2) activate EF
        subproblems.ef.activate()

        # make sure the binaries are converted back if necessary
        if subproblems.relax_binaries: subproblems.unrelax_all_binaries()
        if subproblems.relax_integers: subproblems.unrelax_all_integers()

        # (3) solve the EF with the given max time
        if "gams" in self.solver:
            ef_results = self.opt.solve(subproblems.ef.model, 
                                    add_options = ['$onecho > time.opt',
                                                    'option reslimit ' + str(self.time_ub),
                                                    '$offecho',
                                                    'GAMS_MODEL.optfile=1'],
                                    symbolic_solver_labels=True,
                                    tee = False)

        else:
            ef_results = self.opt.solve(subproblems.ef.model, 
                                    timelimit = self.time_ub,
                                    symbolic_solver_labels=True,
                                    tee = False)
            
        # when we have parallel ranks, because we can have locally optimal solutions
        # which may be initialized differently, make sure to select the BEST solution
        if (size > 1):

            # grab the viable solution, or return -inf (if no solution at this rank)
            if ef_results.solver.termination_condition==TerminationCondition.optimal and \
                    ef_results.solver.status==SolverStatus.ok:
                for obj in subproblems.ef.model.component_objects(pyo.Objective): break
                obj = pyo.value(obj)
            else:
                obj = float("inf")

            # check that there is a viable solution; otw return False
            sol_found = (ef_results.solver.termination_condition==TerminationCondition.optimal \
                            and ef_results.solver.status==SolverStatus.ok \
                                and obj not in self.best_objs)
            num_sols = MPI.COMM_WORLD.allreduce(sol_found, op = MPI.SUM)
            if (num_sols == 0):
                subproblems.ef.deactivate()
                return False, None, math.nan
            
            # determine which objective is best
            best_obj, best_rank = MPI.COMM_WORLD.allreduce([obj, rank],
                                                           op = MPI.MINLOC)
            self.best_objs.append(best_obj)

            # pull the solution for the rank/subproblems and generate the dictionary
            sol = {}
            var_val = None
            for lifted_var_id in subproblems.ef.lifted_var_ids:
                if (rank==best_rank):
                    var_val = pyo.value(subproblems.ef.model.lifted_vars[lifted_var_id])
                
                var_val = MPI.COMM_WORLD.bcast(obj = var_val,
                                               root = best_rank)
                sol[lifted_var_id] = var_val
            
            # return info
            subproblems.ef.deactivate()
            return True, sol, best_obj

        # if we have successfully generated a candidate solution
        if ef_results.solver.termination_condition==TerminationCondition.optimal and \
            ef_results.solver.status==SolverStatus.ok:
                
                # (4) deactivate EF
                for obj in subproblems.ef.model.component_objects(pyo.Objective):
                    subproblems.ef.deactivate()
                    return True, subproblems.ef.save_solution(), pyo.value(obj)

        else:
            # (4) deactivate EF
            subproblems.ef.deactivate()
            return False, None, math.nan