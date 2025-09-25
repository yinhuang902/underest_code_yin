"""
Utility methods for supporting the EF capabilities within SNoGloDe.
"""
import pyomo.environ as pyo
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

from snoglode.components.subproblems import Subproblems, SupportedVars

class ExtensiveForm():
    """
    Class that constructs the extensive form based on the scenarios
    and identified first stage variables.
    """

    def __init__(self, 
                 subproblems: Subproblems) -> None:
        """
        Builds the extensive form using the current subproblem models.
        This is built directly on the Subproblems object under the attr name "ef".
        Modeled this heavily off of how mpi-sppy performs this.

        NOTE (1): this currently just links to the subproblem instances
        i.e. this is NOT a standalone Pyomo model - be careful to activate/deactivate
        nonants / objectives appropriately within the algorithm!!!

        NOTE (2): Lifted vars are found via self.model.lifted_vars (which is 
        an indexed pyo.Var, where each index corresponds to a particular lifted_var_id)
        These lifted vars have domains of pyo.Any; this may be changed in the future.

        NOTE (3): Nonant can be accessed via self.model.nonants. 

        Parameters
        ----------
        scenarios : Subproblems
            Current *initialized* Subproblems object.
        """

        # add the pyomo model of EF, with a block for each subproblem.
        self.model = pyo.ConcreteModel("ef")
        self.model.obj = pyo.Objective(expr = 0.0)

        """ Build the EF lifted variables """

        # get all of the variable IDs
        self.lifted_var_ids = []
        for var_type in SupportedVars:
            lifted_var_ids = list(subproblems.root_node_state[var_type].keys())
            if (lifted_var_ids): self.lifted_var_ids += lifted_var_ids

        # create the first stage variables, indexed by the var_ids
        lifted_vars = pyo.Var(self.lifted_var_ids,
                              name = "lifted_vars",
                              domain = pyo.Reals)
        self.model.add_component("lifted_vars", lifted_vars)

        """ Add each scenarios as a block on the EF & build nonants """

        # keep references to all objectives for easy on/off
        self.subproblem_objectives = []

        # generate models/vars for the subproblems NOT present on this rank.
        all_other_subproblem_names = list(set(subproblems.all_names) - set(subproblems.names))
        self.other_subproblems = Subproblems(subproblem_names = subproblems.all_names,
                                             subproblem_creator = subproblems.subproblem_creator,
                                             subset_subproblem_names = all_other_subproblem_names,
                                             use_fbbt = subproblems.use_fbbt,
                                             obbt_solver_name = subproblems.obbt_solver_name,
                                             obbt_solver_opts = subproblems.obbt_solver_opts,
                                             use_obbt = subproblems.use_obbt,
                                             relax_binaries = subproblems.relax_binaries,
                                             relax_integers = subproblems.relax_integers)

        # add nonants constraint list to build on
        self.model.nonants = pyo.Constraint(pyo.Any)

        # add each of the scenario subproblems
        for subproblem_name in subproblems.all_names:

            # IF the subproblem is not on this rank
            if (subproblem_name not in subproblems.names):

                # add the subproblem model as a block to the EF
                self.model.add_component(subproblem_name, 
                                         self.other_subproblems.model[subproblem_name])
                
                # deactivate subproblem objective + add to the running EF objective
                obj = get_active_objective(self.other_subproblems.model[subproblem_name])
                self.subproblem_objectives.append(obj)
                obj.deactivate()
                self.model.obj.expr += obj * self.other_subproblems.probability[subproblem_name]
                
                # check we have deactivated all objectives
                assert len(list(self.other_subproblems.model[subproblem_name].component_data_objects(pyo.Objective, active=True))) == 0, \
                    "Error building EF: should have no other active objective"

                # add the ref to pyo.Var scenario-specific lifted var to dict
                for subproblem_lifted_var in self.other_subproblems.subproblem_lifted_vars[subproblem_name]:

                    # extract var ID
                    _, lifted_var_id, _ = self.other_subproblems.var_to_data[subproblem_lifted_var]

                    # grab the EF specific first stage variables
                    ef_lifted_var = self.model.lifted_vars[lifted_var_id]

                    # add nonant for scen / first stage var to list
                    self.model.nonants[subproblem_name, lifted_var_id] = \
                        (ef_lifted_var == subproblem_lifted_var)

            # OTW, add directly
            if (subproblem_name in subproblems.names):

                # add the subproblem model as a block to the EF
                self.model.add_component(subproblem_name, 
                                         subproblems.model[subproblem_name])
            
                # deactivate subproblem objective + add to the running EF objective
                obj = get_active_objective(subproblems.model[subproblem_name])
                self.subproblem_objectives.append(obj)
                obj.deactivate()
                self.model.obj.expr += obj * subproblems.probability[subproblem_name]

                # check we have deactivated all objectives
                assert len(list(subproblems.model[subproblem_name].component_data_objects(pyo.Objective, active=True))) == 0, \
                    "Error building EF: should have no other active objective"

                # add the ref to pyo.Var scenario-specific lifted var to dict
                for subproblem_lifted_var in subproblems.subproblem_lifted_vars[subproblem_name]:

                    # extract var ID
                    _, lifted_var_id, _ = subproblems.var_to_data[subproblem_lifted_var]

                    # grab the EF specific first stage variables
                    ef_lifted_var = self.model.lifted_vars[lifted_var_id]

                    # add nonant for scen / first stage var to list
                    self.model.nonants[subproblem_name, lifted_var_id] = \
                        (ef_lifted_var == subproblem_lifted_var)

        # deactivate EF
        self.deactivate()
        if subproblems.verbose: print("EF built.")


    def activate(self):
        """
        Activates the extensive form by:
            (1) deactivating subproblem specific objectives
            (2) activating EF objective
            (3) activating block of nonant constraints
        """
        for obj in self.subproblem_objectives:   # (1)
            obj.deactivate()
        self.model.obj.activate()                # (2)
        self.model.nonants.activate()            # (3)

    
    def deactivate(self):
        """
        Deactivates the extensive form by:
            (1) activating subproblem specific objectives
            (2) deactivating EF objective
            (3) deactivating block of nonant constraints
        """
        for obj in self.subproblem_objectives:    # (1)
            obj.activate()
        self.model.obj.deactivate()               # (2)
        self.model.nonants.deactivate()           # (3)

    
    def save_solution(self) -> dict:
        """
        Takes the current solution and saves it to a dictionary.

        Returns
        ----------
        candidate_solution_state : dict
            dict of keys which correspond to the lifted variable ID
            and the element is the value of that lifted variable ID.
        """
        candidate_solution_state = {}
    
        # for each of the first stage variables, save value
        for lifted_var_id in self.lifted_var_ids:

            # grab the first stage var ref
            ef_lifted_var = self.model.lifted_vars[lifted_var_id]
            
            # grab value, save under var_id
            candidate_solution_state[lifted_var_id] = pyo.value(ef_lifted_var)
        
        return candidate_solution_state
