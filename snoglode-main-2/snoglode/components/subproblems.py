"""
Manages the scenarios (this is where the models are organized / stored)
"""
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.alternative_solutions.obbt import obbt_analysis
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

from snoglode.utils.supported import SupportedVars
from snoglode.components.node import Node
import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# suppress warnings when loading infeasible models
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
import math
import itertools

class Subproblems():
    """
    Class for organizing the subproblems.      
    """

    def __init__(self, 
                 subproblem_names: list, 
                 subproblem_creator: callable, 
                 use_fbbt: bool,
                 use_obbt: bool,
                 obbt_solver_name: str,
                 obbt_solver_opts: dict,
                 relax_binaries: bool,
                 relax_integers: bool,
                 subset_subproblem_names: list = [], 
                 verbose: bool = False) -> None:
        """
        Takes the inputs and generates the models we need to represent each 
        subproblem along with identification of the lifted variables.

        Parameters
        ----------
        subproblem_names : list(str)
            list of string names of subproblems that is len() of num subproblems
        subproblem_creator : function
            given subproblems number, returns
             (1) the subproblem specific pyomo model
             (2) the dictonary of key pairs (varID, pyo.Var)
             (3) the list of subproblem specific pyo.Var's
             (4) the subproblem probability
        use_fbbt : bool
            if FBBT is performed on the subproblems. 
        use_fbbt : bool
            if OBBT is performed on the subproblems.
        obbt_solver : pyo.SolverFactory
            solver for solving OBBT solves; None if not using.
        subset_subproblem_names : list(str) 
                [ WHERE: subset_subproblem_names ⊆ subproblem_names ]
            a subset of all the subproblem_names that are to be initialized
            by this class (can be used by the user (*CAREFULLY) but was 
            added for easier use within the auto EF build)
        relax_binaries : bool, optional
            indicates whether the binary variables in the tree should 
            have their integrality constraints relaxed or not (not required).
        relax_integers : bool, optional
            indicates whether the integers variables in the tree should 
            have their integrality constraints relaxed or not (not required).
        verbose : bool, optional
            if we wish to print updates to the terminal
            default, False
        
        Returns
        --------
        node : Node
            root node of the spatial branch and bound tree.
        """
        # assign info to the object we want to keep
        assert type(subproblem_names)==list, \
            "subproblem names should be a string list of names."
        
        for name in subproblem_names:
            assert isinstance(name, str), \
                "Each scenario name should be a string.\n" + \
                f"\tFirst error found: {name} is of type {type(name)}."
        
        assert len(subproblem_names) >= size, \
            f"number of subproblems({len(subproblem_names)} cannot be less than \
                the number of parallel processes ({size}))"
        
        self.all_names = subproblem_names
        self.verbose = verbose
        self.subproblem_creator = subproblem_creator
        self.use_fbbt = use_fbbt
        self.use_obbt = use_obbt
        self.obbt_solver_name = obbt_solver_name
        self.obbt_solver_opts = obbt_solver_opts
        self.relax_binaries = relax_binaries
        self.relax_integers = relax_integers

        # check that we do not have too many paralell processes
        assert len(subproblem_names) >= size, \
            f"There are {len(subproblem_names)} subproblems and {size} parallel processes.\n" + \
            f"Num. parallel processes should be >= number of subproblems; please re-evaluate."

        # decide which subproblems need to be generated on this rank
        if (len(subset_subproblem_names)): self.names = subset_subproblem_names
        else: self._organize_names_by_rank()

        # generate the subproblems & organize via the dictionary
        if self.verbose: print("Generating the models for the subproblems.")

        # info the subproblems need to track
        self.model = {subproblem_name: None for subproblem_name in self.names}
        self.probability = {subproblem_name: None for subproblem_name in self.names}
        self.subproblem_specific_vars = {subproblem_name: {} for subproblem_name in self.names}
        self.subproblem_lifted_vars = {subproblem_name: [] for subproblem_name in self.names}
        self.id_to_vars = {}                    # var_id: [pyo.Var] all pyomo vars associated with this ID
        self.var_to_data = pyo.ComponentMap()   # pyo.Var: (domain: SupportedVar, var_id: str, subproblem_name: str)

        # building the dictionary to init the root node
        root_node_state = {var_type: {} for var_type in SupportedVars}
        
        # keep track of the # of lifted vars in each generated scenario --> sanity checks
        for subproblem_name in self.names:

            # generate scenario model + list of first stage variables
            subproblem_model, subproblem_lifted_variables, subproblem_probability \
                    = subproblem_creator(subproblem_name)

            # check we have the right types
            assert type(subproblem_lifted_variables)==dict, \
                "second return of subproblem_creator must be a dict of (lifted var id) : pyo.Var."
            assert len(subproblem_lifted_variables.keys())!=0, \
                "must be at least one lifted var element."
            
            # only have one objective!
            num_objs = 0
            for _ in subproblem_model.component_data_objects(pyo.Objective, active=True): num_objs+=1
            assert num_objs==1, "Each subproblem should only have one ACTIVE objective."
            
            if self.verbose: print(f"\tScenario {subproblem_name} has {len(subproblem_lifted_variables.keys())} first stage vars.")

            # perform optional bounds tightening.
            if self.use_fbbt: self._perform_fbbt(subproblem_model)
            if self.use_obbt: self._perform_obbt(model = subproblem_model,
                                                 variables = self.subproblem_lifted_vars[subproblem_name])

            # INITIALIZING THE LIFTED VARIABLES -> for branching
            for lifted_var_id in subproblem_lifted_variables:

                # update information in the dict.
                self._init_lifted_var(node_state = root_node_state,
                                      lifted_var_id = lifted_var_id,
                                      lifted_var = subproblem_lifted_variables[lifted_var_id],
                                      subproblem_name = subproblem_name)
            
            # INITIALIZING THE SUBPROBLEM SPECIFIC VARIABLES -> for bounding
            for var in subproblem_model.component_data_objects(pyo.Var):
                
                # if this is a complicating variable, skip
                if var not in self.var_to_data:

                    # if it is indexed, flatten it
                    if var.is_indexed():
                        for i in var._index_set:
                            
                            # add to dict
                            self._init_subproblem_specific_var(subproblem_specific_var = var[i],
                                                               subproblem_name = subproblem_name)
                            
                    # if it is not, send as is
                    else:
                        # add to dict
                        self._init_subproblem_specific_var(subproblem_specific_var = var,
                                                        subproblem_name = subproblem_name)

            # init the LB objective cut
            self._init_bound_cut(subproblem_model)
            
            # name the Pyomo model with the current scenario name
            subproblem_model.name = subproblem_name

            # since we passed, add info
            self.model[subproblem_name] = subproblem_model
            self.probability[subproblem_name] = subproblem_probability
        
        if self.verbose: print("Finished generating subproblem models.")

        # accumulate all of the lifted variables dictionary across ranks
        self._sync_all_lifted_vars(root_node_state)
        self.root_node_state = root_node_state

        # if we relax binaries, save a list of those var ID's
        if self.relax_binaries and len(self.root_node_state[SupportedVars.binary]): 
            self.binary_var_ids = [lifted_var for lifted_var in self.root_node_state[SupportedVars.binary]]
        
        # if we relax integers, save a list of those var ID's
        if self.relax_integers:
            if self.root_node_state[SupportedVars.integers]: 
                self.integer_var_ids = [lifted_var for lifted_var in self.root_node_state[SupportedVars.integers]]
            if self.root_node_state[SupportedVars.nonnegative_integers]:
                self.nonneg_integer_var_ids = [lifted_var for lifted_var in self.root_node_state[SupportedVars.nonnegative_integers]]
        
        # extract the list of all variable ids
        self.lifted_var_ids = [lifted_var for var_domain in SupportedVars\
                                    for lifted_var in self.root_node_state[var_domain]]


    def set_all_states(self,
                       state: dict,
                       set_second_stage: bool = True) -> None:
        """
        Sets the state of ALL scenario models based on the node passed.
        This is used to help with the bounding problems (i.e,. EF)

        Parameters
        ----------
        subproblems : snoglode.components.Subproblems
            istantiated Subproblems object.
        state : Node.state
            dictionary corresponding to current node state.
        set_second_stage: bool
            indicates if we should reset the seconds the second stage
            variables to their bounds, or not.
        """
        # for each scenario's model
        for subproblem_name in self.names:

            # set the current state (i.e., var bounds/fixing to reflect current spacial area)
            self.set_subproblem_state(subproblem_name = subproblem_name,
                                      state = state,
                                      set_second_stage = set_second_stage)


    def set_subproblem_state(self, 
                             subproblem_name: str,
                             state: dict,
                             set_second_stage: bool) -> None:
        """
        Given the first stage variables and the state dictionary
        set the required bounds / fixed states of the first stage variables.

        Parameters
        -----------
        subproblem_name : str
            name of the subproblem
        state : dict
            in the spacial BnB tree we are in. It describes the bounds
            and fixed states of all variables.
        set_second_stage: bool
            indicates if we should reset the seconds the second stage
            variables to their bounds, or not.
        """
        # for each of the lifted variables
        for lifted_var in self.subproblem_lifted_vars[subproblem_name]:

            # extract lifted var ID, domain
            var_type, lifted_var_id, subproblem_name = self.var_to_data[lifted_var]

            # update variables state to reflect node
            self._update_lifted_var_state(subproblem_name = subproblem_name,
                                          lifted_var = lifted_var,
                                          lifted_var_state = state[var_type][lifted_var_id])

        
        # if we choose to do any bounds tightening, update the bounds on the subproblem specific vars as well.
        if set_second_stage:
            for subproblem_var_name in self.subproblem_specific_vars[subproblem_name]:

                # retrieve pyo.Var object
                subproblem_var = self.subproblem_specific_vars[subproblem_name][subproblem_var_name]["var"]

                # set the lb / ub
                subproblem_var.lb = self.subproblem_specific_vars[subproblem_name][subproblem_var_name]["lb"]
                subproblem_var.ub = self.subproblem_specific_vars[subproblem_name][subproblem_var_name]["ub"]

                # if it was fixed, re-fix to the proper value
                if self.subproblem_specific_vars[subproblem_name][subproblem_var_name]["fixed"]:
                    subproblem_var.fix(self.subproblem_specific_vars[subproblem_name][subproblem_var_name]["value"])

                # otw, unfix.
                else: 
                    subproblem_var.unfix()
    

    def tighten_and_sync_bounds(self, node) -> None:
        """
        Once we have performed bounds tightening across all subproblems,
        we would like to propogate all of the bounds across the subproblems.

        Example:
            Consider x is bounded between (0,1)
            We have 3 subproblems, s \in S = [1,2,3]

            After performing some type of bounds tightening on the the 
            individual models, we find that:
                - x_1 is bounded between (0.1, 0.7)
                - x_2 is bounded between (0.15, 0.8)
                - x_3 is bounded between (0.3, 0.9)
            
            Because we know these are the reduced feasible regions
            for these variables, we can sync these bounds to further
            inform/tightening the LB relaxation.

            In this case:
                x_LB = max(x_1.lb, x_2.lb, x_3.lb) = max(0.1, 0.15, 0.3) = 0.3
                x_UB = min(x_1.ub, x_2.ub, x_3.ub) = min(0.7, 0.8, 0.9) = 0.7

            So the new bounds, across all subproblems should be (0.3, 0.7)
        
        Additionally, if this is a parallel run, we need to sync
        across all the subproblems on alternative.

        Parameters
        ----------
        node : Node
            current node, from which we will we update the bounds.
        """
        if not self.use_fbbt and not self.use_obbt: return True

        # (1) perform FBBT and/or OBBT on all subproblems on this rank
        # NOTE: this tightens and syncs bounds across subproblems
        feasible = self._tighten_rank_subproblem_bounds(node)
        if not feasible: return False

        # (2) if in parallel, determine new bounds for all subproblems on all ranks
        # update the node
        self._sync_all_lifted_vars(node.state)

        # (3) set new bounds on all subproblems
        # NOTE: don't reset the second stage this time (modified in place)
        self.set_all_states(node.state,
                            set_second_stage = False)
        return True


    def relax_all_binaries(self) -> None:
        """
        Takes all binaries with the domain SupportVars.binary
        and converts their domains to Reals bounded from (0,1)
        """
        # should only be using this functionality if we indicated relaxation
        assert self.relax_binaries

        # for each of the variables that is supposed to be a relaxed binary
        for var_ID in self.binary_var_ids:
            for var in self.id_to_vars[var_ID]:
                self._relax_binary(var)
    

    def relax_all_integers(self) -> None:
        """
        Takes all integers with the domain SupportVars.binary
        and converts their domains to Reals bounded from their original bounds
        """
        # should only be using this functionality if we indicated relaxation
        assert self.relax_integers

        # for each of the variables that is supposed to be a relaxed integer
        if hasattr(self, "integer_var_ids"):
            for var_ID in self.integer_var_ids:
                for var in self.id_to_vars[var_ID]:
                    self._relax_integer(var=var,
                                        lb = var.lb,
                                        ub = var.ub)
        if hasattr(self, "nonneg_integer_var_ids"):
            for var_ID in self.nonneg_integer_var_ids:
                for var in self.id_to_vars[var_ID]:
                    self._relax_integer(var=var,
                                        lb = var.lb,
                                        ub = var.ub)


    def unrelax_all_binaries(self) -> None:
        """
        Takes all binaries with the domain SupportVars.binary
        and converts their domains back to binary
        """
        # should only be using this functionality if we indicated relaxation
        assert self.relax_binaries

        # for each of the variables that is supposed to be a relaxed binary
        for var_ID in self.binary_var_ids:
            for var in self.id_to_vars[var_ID]:
                self._convert_to_binary(var)
    

    def unrelax_all_integers(self) -> None:
        """
        Takes all integers with the domain SupportVars.intergers/nonnegative_integers
        and converts their domains back to their respective integer domains
        """
        # should only be using this functionality if we indicated relaxation
        assert self.relax_integers

        if hasattr(self, "integer_var_ids"):
            for var_ID in self.integer_var_ids:
                for var in self.id_to_vars[var_ID]:
                    self._convert_to_integer(var)
        if hasattr(self, "nonneg_integer_var_ids"):
            for var_ID in self.nonneg_integer_var_ids:
                for var in self.id_to_vars[var_ID]:
                    self._convert_to_nonneg_integer(var)


    def save_results_to_dict(self) -> dict:
        """
        Saves the values of ALL variables at each of the models
        for each of the scenarios. 
        
        Returns a dict of the results.
        """
        # format is [scenario name][variable name] -> variable value
        results = {subproblem_name: {} for subproblem_name in self.names}
        
        # go through each solved subproblem model
        for subproblem_name in self.names:
            subproblem_model = self.model[subproblem_name]

            for var in subproblem_model.component_objects(pyo.Var):

                # if we are indexed, unpack
                if var.is_indexed():
                    
                    for k in var.keys():
                        try: results[subproblem_name][var[k].name] = pyo.value(var[k], exception=False)
                        except ValueError: results[subproblem_name][var[k].name] = None
                
                # otherwise add directly
                else:
                    try: results[subproblem_name][var.name] = pyo.value(var, exception=False)
                    except ValueError: results[subproblem_name][var.name] = None
        
        return results
    

    def _init_lifted_var(self, 
                         node_state: dict,
                         lifted_var_id: str, 
                         lifted_var: pyo.Var,
                         subproblem_name: str) -> None:
        """
        Given a pyomo variable (identified as lifted)
        Record relevant information for branching purposes.

            - Continuous Vars -> bounds
            - Binary Vars -> fixed / value

        Parameters
        -----------
        lifted_var_id : hashable type
            lifted pyomo variable id value.
        lifted_var : pyo.Var
            lifted pyomo variable.
        subproblem_name : str
            subproblem string name representation corresponding to this subproblem.
        """

        # should not have any indexed vars...
        assert not lifted_var.is_indexed(), \
            f"{lifted_var.name} was passed as an indexed Var. \n \
                Every lifted variable should be passed as a single element, not indexed. "
        
        # to use branching, we need to have bounds on all lifted vars
        assert lifted_var.has_lb() and lifted_var.has_ub(), \
            f"{lifted_var.name} was passed without bounds. \n \
                Every lifted variable must have bounds to perform branching. "

        # extract domain and convert to associated support variable domain
        var_type = SupportedVars(lifted_var.domain.local_name)
        
        # add var to ComponentMap
        self.var_to_data[lifted_var] = (var_type, lifted_var_id, subproblem_name)

        # if we have not entered this variable already, add it now
        if lifted_var_id not in self.id_to_vars:

            # add to ID -> [vars]
            self.id_to_vars[lifted_var_id] = [lifted_var]

            # add to the node state
            node_state[var_type][lifted_var_id] = \
                        LiftedVariable(domain = var_type,
                                       lb = lifted_var.lb,
                                       ub = lifted_var.ub,
                                       var_id = lifted_var_id)
            
        # if we have already entered this variable, update it.
        else:
            # add to ID -> [vars]
            self.id_to_vars[lifted_var_id].append(lifted_var)

            # update on the node state (saves best bounds)
            node_state[var_type][lifted_var_id].update(lifted_var)
            
        # add to list of lifted variables for this subproblem
        self.subproblem_lifted_vars[subproblem_name].append(lifted_var)


    def _init_subproblem_specific_var(self, 
                                      subproblem_specific_var: pyo.Var, 
                                      subproblem_name: str) -> None:
        """
        Given a pyomo variable (identified as subproblem specific)
        Record relevant information for branching.

        Parameters
        -----------
        subproblem_specific_var : pyo.Var
            subproblem specific pyomo variable.
        subproblem_name : str
            subproblem string name representation corresponding to this subproblem.
        """
        # check edge case - do we have this name already in this dict for this subproblem?
        assert subproblem_specific_var.name not in self.subproblem_specific_vars[subproblem_name], \
            "cannot have two local variables with the same name.\n" + \
            f"at subproblem = {subproblem_name}, variable name {subproblem_specific_var.name} appears at least twice."

        lb = None if not subproblem_specific_var.has_lb() else subproblem_specific_var.lb
        ub = None if not subproblem_specific_var.has_ub() else subproblem_specific_var.ub
        self.subproblem_specific_vars[subproblem_name][subproblem_specific_var.name] = {
                "lb": lb,
                "ub": ub,
                "fixed": subproblem_specific_var.is_fixed(),
                "value": subproblem_specific_var.value,
                "var": subproblem_specific_var
            }


    def _init_bound_cut(self,
                        subproblem_model: pyo.ConcreteModel):
        """
        At the LB problem, we can bound the objective
        by the successor nodes objective at that subproblem.
        """
        # dummy successor obj for now
        subproblem_model.successor_obj = pyo.Param(initialize = float("-inf"),
                                                   mutable = True)

        # add the constraint on the objective term
        obj = get_active_objective(subproblem_model)
        subproblem_model.successor_lb_cut = pyo.Constraint( expr = obj >= subproblem_model.successor_obj )
        subproblem_model.successor_lb_cut.deactivate()


    def _organize_names_by_rank(self) -> None:
        """
        If we are NOT given names for each subproblem to be built on a rank,
        assign subproblems to ranks by a simple slicing method.
        """

        # determine how many problems per rank
        num_subproblems_per_rank = np.floor(len(self.all_names) / size)
        num_residual_subproblems = len(self.all_names) - (num_subproblems_per_rank * size)

        # add the residual subproblems to the first few ranks
        if (rank <= (num_residual_subproblems-1)):
            start_index = (num_subproblems_per_rank + 1) * rank
            end_index = start_index + (num_subproblems_per_rank + 1)
        else:
            # start at the end_index of ranks w/ residuals, and add from there.
            start_index = (num_subproblems_per_rank + 1) * (num_residual_subproblems - 1) \
                            + (num_subproblems_per_rank + 1) \
                                + (num_subproblems_per_rank * (rank - num_residual_subproblems))
            end_index = start_index + num_subproblems_per_rank

        self.names = self.all_names[int(start_index) : int(end_index)]


    def _sync_all_lifted_vars(self,
                              node_state: dict) -> None:
        """
        After we have initialized all of the lifted variables across 
        subproblems on all ranks, we need to share all relevant information
        for branching / tree management purposes.

        Generally, we need to make sure
        we are selecting the proper bounds -> some subproblems may have
        different bounds on the same lifted variable. Make sure this is
        corrected by broadcasting back out.

        Update node bounds to best new state.

        Parameters
        ----------
        node : Node, optional
            current node in the BnB tree
        update_var : bool, optional
            indicates if we should also update the var objects obunds
            to reflect the bound change.
            By default, False
        """
        if (size > 1):
            MPI.COMM_WORLD.barrier()
            
            # based on how we stored the lifted vars, go through each of the types and
            # gather all of the lifted_var_ids on each rank associated with this
            for var_domain in SupportedVars:

                # determine the number of variables we have for this domain
                num_lifted_vars = len(node_state[var_domain])
                total_num_lifted_vars = MPI.COMM_WORLD.allreduce(num_lifted_vars, op=MPI.SUM)

                # only perform sync operations if there exists lifted variables for this domain
                if (total_num_lifted_vars > 0):

                    # gather all names of lifted variables from all ranks
                    lifted_vars = list(lifted_var_id for lifted_var_id in node_state[var_domain])
                    all_lifted_vars = MPI.COMM_WORLD.allgather(lifted_vars)

                    # flatten list (itertools) & take the set to remove duplicates
                    set_of_lifted_vars = sorted(set(itertools.chain.from_iterable(all_lifted_vars)))

                    # sync all names / bounds on the current dict
                    for lifted_var_id in set_of_lifted_vars:

                        # if this rank does not have it, add
                        if lifted_var_id not in node_state[var_domain]:
                            node_state[var_domain][lifted_var_id] = \
                                LiftedVariable(domain = var_domain,
                                               lb = -math.inf,
                                               ub = math.inf,
                                               var_id = "NAN")
                        
                        MPI.COMM_WORLD.barrier()
                        
                        # save the best bounds on all ranks
                        best_lb = MPI.COMM_WORLD.allreduce(node_state[var_domain][lifted_var_id].lb, op=MPI.MAX)
                        best_ub = MPI.COMM_WORLD.allreduce(node_state[var_domain][lifted_var_id].ub, op=MPI.MIN)

                        # update to best bounds
                        node_state[var_domain][lifted_var_id].lb = best_lb
                        node_state[var_domain][lifted_var_id].ub = best_ub


    def _update_lifted_var_state(self,
                                 subproblem_name: str,
                                 lifted_var: pyo.Var, 
                                 lifted_var_state: dict) -> None:
        """
        Takes a variable Pyomo object, and updates the passed criteria
        to reflect the current node state.

        Parameters
        ----------
        lifted_var : pyo.Var
            the Pyomo variable object (not indexed)
        lifted_var_state : dict
            information reflecting the variables current state
            that we are to update to.
        """
        # retrieve fixed state status & update
        if lifted_var_state.is_fixed: lifted_var.fix(lifted_var_state.value)
        else: lifted_var.unfix()

        # update bounds
        assert lifted_var_state.lb <= lifted_var_state.ub
        lifted_var.lb = lifted_var_state.lb
        lifted_var.ub = lifted_var_state.ub

    
    def _perform_fbbt(self, 
                      model: pyo.ConcreteModel) -> pyo.ComponentMap:
        """
        Performs FBBT, from Pyomo, to the given model.
        Applies this to ALL of the variables.
            (first and second stage)

        Parameters
        -----------
        subproblem_name : str
            name corresponding to the subproblem we want to perform
            fbbt on.
        """
        # perform fbbt, update bounds to make obbt more efficient
        # and makes sure to set the second stage vars
        try: 
            variable_ranges = fbbt(model)

            # check the bounds, fix if possible, otw infeasibility will be caught in model
            feasible = self._check_bounds(variable_ranges)
            if not feasible: return False

        except InfeasibleConstraintException:
            return False
    
        return variable_ranges

    
    def _perform_obbt(self, 
                      model: pyo.ConcreteModel,
                      variables: list,
                      tol: float = 1e-1) -> pyo.ComponentMap:
        """
        Performs OBBT, from Pyomo, to the given model.
        Applies this to ONLY FIRST STAGE variables.
            -> more expensive than fbbt

        Parameters
        -----------
        model : pyo.ConcreteModel
            Model that we aim to perform OBBT on.
        variables : list
            All variables we want to perform OBBT on within model.
        tol : float
            anything less than this relative change, stop.
        """ 
        try:
            # compute new ranges for first stage vars
            variable_ranges = obbt_analysis(model = model,
                                            solver = self.obbt_solver_name,
                                            solver_options = self.obbt_solver_opts,
                                            variables = variables,
                                            warmstart = False)
            model.del_component(model._obbt)
            
            # set new values
            for var in variables:
                new_lb, new_ub = variable_ranges[var]
                var.lb = new_lb
                var.ub = new_ub

            # perform again and evaluate the tolerance
            tol_met = False
            while (not tol_met):
                
                # compute new ranges for first stage vars
                variable_ranges_update = obbt_analysis(model = model,
                                                    solver = self.obbt_solver_name,
                                                    solver_options = self.obbt_solver_opts,
                                                    variables = variables,
                                                    warmstart = False)
                model.del_component(model._obbt)
                
                # set new bounds
                for var in variables:
                    new_lb, new_ub = variable_ranges_update[var]
                    var.lb = new_lb
                    var.ub = new_ub
                
                # check if we have met the tolerance for all of the bounds
                tol_met = True
                for var in variables:
                    old_lb, old_ub = variable_ranges[var]
                    new_lb, new_ub = variable_ranges_update[var]
                    if (old_lb - new_lb > tol) or (old_ub - new_ub > tol): 
                        tol_met = False
                        variable_ranges = variable_ranges_update
                        break
                        
            feasible = self._check_bounds(variable_ranges)
            if not feasible: return False

        except RuntimeError:
            # sometimes, we raise RuntimeError: OBBT cannot be applied, TerminationCondition = infeasible
            # in this case, model is infeasible.
                return False

        return variable_ranges


    def _tighten_rank_subproblem_bounds(self,
                                        node: Node) -> bool:
        """
        For each subproblem on this rank, perform whichever bounds
        tightening procedures were indicated.

        Updates the saved bounds on the variables in place.

        What happens if we have binaries?

        Parameters
        ----------
        node : Node
            node representing the current feasible range

        Returns
        --------
        feasible : bool
            if the bounds tightening indicates an infeasible problem
            return False, otherwise return True.
        """
        assert (self.use_fbbt) or (self.use_obbt), \
            "shouldn't call tighten_all_bounds if there is no \
            bounding methods indicated"
        
        # for each subproblem, tighten bounds
        for subproblem_name in self.names:

            # extract model, list of flattened first stage vars
            model = self.model[subproblem_name]

            # only care about variables that can still be branched on
            # for each of the lifted variables within this subproblem
            variables = []
            for var in self.subproblem_lifted_vars[subproblem_name]:
                domain, varID, _ = self.var_to_data[var]
                
                # if this variable ID is still within the to_branch, add to the running list
                if varID in node.to_branch[domain]:
                    variables.append(var)

            # perform FBBT first - updates variable bounds in place
            if self.use_fbbt:
                fbbt_bounds = self._perform_fbbt(model)

            # perform OBBT afterwards
            if self.use_obbt:
                obbt_bounds = self._perform_obbt(model = model,
                                                 variables = variables)
            
            # update bounds on the first stage variables
            for var in variables:
                max_bounds = (float("-inf"), float("inf"))

                # extract bounds (if we computed them- otw max)
                if (not self.use_fbbt) or (fbbt_bounds == False):
                    lb_fbbt, ub_fbbt = max_bounds
                else: lb_fbbt, ub_fbbt = fbbt_bounds[var]
                
                if (not self.use_obbt) or (obbt_bounds == False):
                    lb_obbt, ub_obbt = max_bounds
                else: lb_obbt, ub_obbt = obbt_bounds[var]

                # find which variable ID / domain / bounds this first stage variable corresponds to
                var_domain, var_id, _ = self.var_to_data[var]
                lb_best = node.state[var_domain][var_id].lb
                ub_best = node.state[var_domain][var_id].ub

                node.state[var_domain][var_id].lb = max(lb_fbbt, lb_obbt, lb_best)
                node.state[var_domain][var_id].ub = min(ub_fbbt, ub_obbt, ub_best)
        
        return True
        

    def _check_bounds(self, 
                      variable_bounds: pyo.ComponentMap) -> bool:
        """
        Given a model, checks the bounds on the variables based on what 
        type of bounds tightening is specified.

        This is a quick fix - we assert that the delta is significantly
        small because we are really jsut trying to fix numerical intolerance
        issues with overlap at the LB/UB;
        If the assert fails, more work needs to be done to determine
        the issue.

        Parameters
        ----------
        variable_bounds: pyo.ComponentMap
            new bounds as determined by FBBT
        """
        for var, bounds in variable_bounds.items():
            lb = bounds[0]
            ub = bounds[1]

            # if we don't have one of these bounds, it's fine
            if lb == None or ub == None: return True

            if lb > ub:

                delta = lb - ub
                assert (delta <= 1e-8)
                lb = lb - 5*delta
                ub = ub + 5*delta

                # if that did not fix it, assume model infeasible.
                if lb > ub: return False

                # update revised bounds
                variable_bounds[var] = (lb, ub)
        
        # everything's fixed, return.
        return True
    

    def _relax_binary(self, var: pyo.Var) -> None:
        """
        Convert domain from pyo.Binary -> pyo.Reals
        and bound from (0,1)

        Parameters
        ----------
        var : pyo.Var
            binary var
        """
        var.domain = pyo.Reals
        var.lb = 0
        var.ub = 1
    

    def _relax_integer(self, 
                       var: pyo.Var,
                       lb: float,
                       ub: float) -> None:
        """
        Convert domain from pyo.Integer/pyo.NonnegativeIntegers -> pyo.Reals
        and bound from original domain

        Parameters
        ----------
        var : pyo.Var
            integer var 
        """
        var.domain = pyo.Reals
        var.lb = lb
        var.ub = ub
    

    def _convert_to_binary(self, var: pyo.Var) -> None:
        """
        Convert domain from pyo.Reals -> pyo.Binary

        Parameters
        ----------
        var : pyo.Var
            relaxed binary var
        """
        var.domain = pyo.Binary
    

    def _convert_to_integer(self, var: pyo.Var) -> None:
        """
        Convert domain from pyo.Reals -> pyo.Binary

        Parameters
        ----------
        var : pyo.Var
            relaxed binary var
        """
        var.domain = pyo.Integers
    
    def _convert_to_nonneg_integer(self, var: pyo.Var) -> None:
        """
        Convert domain from pyo.Reals -> pyo.Binary

        Parameters
        ----------
        var : pyo.Var
            relaxed binary var
        """
        var.domain = pyo.NonNegativeIntegers


class LiftedVariable():
    """
    This class object stores all of the information related to a 
    lifted variable.

    It is updated to indicate the best bounds, and updating
    which subproblems the variable appears in.
    """
    def __init__(self,
                 domain: SupportedVars,
                 lb: float,
                 ub: float,
                 var_id: str,
                 fixed: bool = False,
                 value: float = None) -> None:
        """
        creates a new lifted variable object.

        Parameters
        -----------
        domain: SupportedVars
            indicates which domain this variable belongs to
        var: pyo.Var
            the pyomo variable representing this variable
        var_id: str
            ID corresponding to this variable
        subproblem_name: str
            name corresponding to the subproblem this variable appears in
        fixed: bool, optional
            indicates if the variable is fixed
        value: float
            indicates what value the fixed variable should be.
        """
        self.domain = domain
        self.lb = lb
        self.ub = ub
        self.is_fixed = fixed
        self.value = value
        self.var_id = var_id
    
    def __name__(self):
        return self.var_id

    def update(self, 
               var: pyo.Var):
        """
        updates a currently instantiated lifted variable
        for a continuous variable, save the best bound.

        add the subproblem the var appears in to the list.

        Parameters
        -----------
        var: pyo.Var
            the pyomo variable representing this variable
        subproblem_name: str
            name corresponding to the subproblem this variable appears in
        """
        
        # need to update to best bounds (if a continuous)
        if self.domain == SupportedVars.reals:
            self.lb = max(self.lb, var.lb)
            self.ub = min(self.ub, var.ub)


class DummyLiftedVariable():
    """
    When we do not have a lifted variable present on a current
    subproblem (because we are not working with a classical 
    structured problem) - create a dummy variable to live in it's place
    so when we do MPI based updates we can use the common methods.
    """
    
    def __init__(self,
                 domain: SupportedVars,
                 lb: float,
                 ub: float):
        self.domain = domain
        self.lb = lb
        self.ub = ub

    def update(self, 
               var: pyo.Var,
               subproblem_name: str):
        raise NameError("The DummyLiftedVariable cannot be updated.")