"""
Rather than passing an excessive number of flags to the solver
initialization, here we will create a single unified object
that holds all of the information necessary.
"""
import pyomo.environ as pyo
from snoglode.bounders.lower_bounders import AbstractLowerBounder, DropNonants
from snoglode.bounders.upper_bounders import AbstractCandidateGenerator, SolveExtensiveForm
from snoglode.components.queues import QueueStrategy
from snoglode.components.branching import SelectionStrategy, PartitionStrategy, RandomSelection, Midpoint

class SolverParameters():
    def __init__(self, 
                 subproblem_names: list,
                 subproblem_creator: callable,
                 lb_solver = pyo.SolverFactory("gurobi"),
                 ub_solver = pyo.SolverFactory("gurobi"),
                 cg_solver = None) -> None:
        """
        The user *must* specify:
            (1) subproblem names
            (2) subproblem creator function
            (3) solvers to be used at UB/LB/CG
        for each of the subproblem solves.

        Parameters
        ----------
        subproblem_names: int or list
            int: how many subproblems we will decompose into
            list: name of subproblems that is len() of num subproblems
        subproblem_creator: callable
            given subproblem number, returns the subproblem pyomo model
            and the list of lifted variables.
        lb_solver: SolverFactory
            initialized solver factory to be used for the LOWER bounding
            problem solves
        ub_solver: SolverFactory
            initialized solver factory to be used for the UPPER bounding
            problem solves
        cg_solver : SolverFactory, optional
            initialized solver factory to be used for the CANDIDATE GENERATOR
            problem solves - optional because sometimes the CG doesn't require
            an optimization problem solve.
        """

        assert type(subproblem_names)==int or type(subproblem_names==list), \
            "number of subproblems should be an integer or a list."
        self._subproblem_names = subproblem_names
        
        assert callable(subproblem_creator), \
            "subproblem_creator should be a callable function."
        self._subproblem_creator = subproblem_creator

        assert lb_solver.available(), "LB solver is not reported to be available."
        self._lb_solver = lb_solver

        assert ub_solver.available(), "UB solver is not reported to be available."
        self._ub_solver = ub_solver        
        
        if (cg_solver != None): assert cg_solver.available(), "CG solver is not reported to be available."
        self._cg_solver = cg_solver    

        # optional, set to defaults

        self._global_guarantee = True
        self._rank_subproblem_names = list()
        self._epsilon = 1e-3
        self._fbbt = True
        self._obbt = False
        self._obbt_solver_name = None
        self._obbt_solver_opts = None
        self._relax_binaries = False
        self._relax_integers = False
        # bounder problem information
        self._lower_bounder = DropNonants
        self._candidate_solution_finder = SolveExtensiveForm
        self._queue_strategy = QueueStrategy.bound
        self._node_feasibility_check = None
        # branching information
        self._selection_strategy = RandomSelection
        self._selection_strategy_solver = None
        self._partition_strategy = Midpoint
        # verbosity / logging
        self._verbose = False
        self._log = False
        self._logname = "log"
        self._loglevel = "INFO" # "DEBUG"

        self._inhert_solutions = False


    def guarantee_global_convergence(self) -> None:
        """
        If we want a global guarantee, the upper bound candidate
        solution *must* be re-solved to global optimality at all
        of the subproblems (this is only relevant really if the 
        candidate generator is a local EF solve)

        This has the potential to increase the solution time,
        but it will guarantee that we can reach the global optimum.
        (though, even without this, there is a possibility we reach global)
        """
        self._global_guarantee = True
    

    def deactivate_global_guarantee(self) -> None:
        """
        If we want a global guarantee, the upper bound candidate
        solution *must* be re-solved to global optimality at all
        of the subproblems (this is only relevant really if the 
        candidate generator is a local EF solve)

        This has the potential to increase the solution time,
        but it will guarantee that we can reach the global optimum.
        (though, even without this, there is a possibility we reach global)
        """
        self._global_guarantee = False


    def set_rank_subproblem_names(self,
                                  rank_subproblem_names: list) -> None:
        """
        Automatically rank subproblems are named by number.
        This is an override in case we want more detailed names.

        Parameters
        ----------
        rank_subproblem_names : list
            subset of the initial subproblem_names list that are to be built
            specifically on this rank (if empty, allocation is automatic)
        """
        assert type(rank_subproblem_names) == list,\
            "Needs to be a list of names."  # more checks done within init of Subproblems
        self._rank_subproblem_names = rank_subproblem_names


    def set_epsilon(self,
                    epsilon: float) -> None:
        """
        This parameter determines when to STOP branching on the continuous
        space of all branching variables. It is an ABSOLUTE tolerance.

        Consider a variable that we are branching on between (0,1).
        At a certain point, say ε=0.1, we will not create nodes
        that does not have at last ε between the LB and the UB.

        Ex:
          - (0.1, 0.40) is valid (Δ = 0.4 - 0.1 = 0.3 > ε)
          - (0.1, 0.15) is NOT valid (Δ = 0.15 - 0.1 = 0.05 < ε)

        Parameters
        ----------
        epsilon : float
            epsilon tolerance is used to define when to stop branching on a continuous variable.
        """
        assert type(epsilon) == float, "epsilon parameter should be a float."
        self._epsilon = epsilon
    
    
    def set_bounds_tightening(self, 
                              fbbt: bool = True,
                              obbt: bool = True,
                              obbt_solver_name: str = "gurobi",
                              obbt_solver_opt: dict = {}) -> None:
        """
        If we want a global guarantee, the upper bound candidate
        solution *must* be resolved to global optimality at all
        of the subproblems (this is only relevant really if the 
        candidate generator is a local EF solve)

        Parameters 
        -----------
        fbbt : bool
            whether or not to perform feasibility based bounds tightening.
        obbt : bool
            whether or not to perform optimization based bounds tightening.
        obbt_solver_name : str
            name of the solver to use with obbt
        obbt_solver_opt : dict
            all of the options needed to set in the obbt solver
        """
        assert type(fbbt) == bool
        self._fbbt = fbbt

        assert type(obbt) == bool
        self._obbt = obbt

        assert type(obbt_solver_name) == str
        self._obbt_solver_name = obbt_solver_name

        assert type(obbt_solver_opt) == dict
        self._obbt_solver_opts = obbt_solver_opt
        

    def relax_binaries(self, 
                       relax: bool = True) -> None:
        """
        We may want to further relax the lower bounding problem
        by relaxing the integrality constriants on binary variables.

        If activated, all binaries are relaxed for the LB problem.

        Parameters
        ----------
        relax : bool, optional
            by default, this ensure binaries are relaxed.
        """
        assert type(relax) == bool, "The optional argument should be a boolean."
        self._relax_binaries = relax
    

    def relax_integers(self, 
                       relax: bool = True) -> None:
        """
        We may want to further relax the lower bounding problem
        by relaxing the integrality constriants on integer variables.

        If activated, all integers are relaxed for the LB problem.

        Parameters
        ----------
        relax : bool, optional
            by default, this ensure binaries are relaxed.
        """
        assert type(relax) == bool, "The optional argument should be a boolean."
        self._relax_integers = relax
    

    def set_bounders(self,
                     lower_bounder: AbstractLowerBounder = DropNonants,
                     candidate_solution_finder: AbstractCandidateGenerator \
                        = SolveExtensiveForm) -> None:
        """
        We allow for high levels of customization for the lower bounding
        problem along with the upper bounding problem.

        Both counders can be from the preset list available within
        the solver, or custom made.

        Parameters
        ----------
        lower_bounder : snoglode.lower_bounders
            one of the lower bounding options availabe in the lower_bounders dir.
            Default, DropNonants
        candidate_solution_finder : snoglode.candidate_solution
            one of the candidate solution generation methods in the candidate_solution dir.
            Default, SolveExtensiveForm
        """
        self._lower_bounder = lower_bounder
        self._candidate_solution_finder = candidate_solution_finder
    
    
    def set_queue_strategy(self, 
                           strategy: QueueStrategy) -> None:
        """
        The queing strategy determines how we select the next open 
        node to explore within the tree. 

        See QueueStrategy object to see all options available.

        Parameters
        ----------
        queue_strategy : QueueStrategy
            Indicates what queing strategy should be used for selecting the next node
            off of the BnB tree.
        """
        self._queue_strategy = strategy
    

    def add_node_feasibility_checker(self,
                                     feasibility_checker: callable) -> None:
        """
        For some problems, we can easily and efficiently check
        if a certain node is feasible without having to initialize
        and solver the LB problem at all.

        The function should take a node object as input
        and it will return True if feasible and False if infeasible.

        CAREFUL - if a node is marked as infeasible that IS feasible,
        it destroys the sanctity of the algorithm.

        Parameters
        ----------
        node_feasibility_check : callable function
            takes in a Node within the sBnB tree & determines if the current
            node is trivially infeasible based on a specific problem (danger zone)
        """
        assert callable(feasibility_checker)
        self._node_feasibility_check = feasibility_checker


    def set_branching(self,
                      selection_strategy: SelectionStrategy = RandomSelection,
                      selection_strategy_solver = None,
                      partition_strategy: PartitionStrategy = Midpoint) -> None:
        """
        There are two major decisions to be made when branching:
            (1) which variable should be selected?
            (2) where should we partition the bounds of that (cont) variable?
        This can be a user specified class, or one of the defaults available in snoglode.

        Parameters
        ----------
        selection_strategy: SelectionStrategy, optional
            The method we will call that can specify which variable to 
            branch on next; by default, random selection.
        selection_strategy_solver: pyo.SolverFactory, optional
            Some methods (strong branching) require a solver.
            Specify the SolverFactory solver we wish to use.
            By default None, since most strategies do not need.
        partition_strategy: PartitionStrategy
            The method we will call that determines, based on the variable
            selected for branching, where to split the current domain
            by default, split at the midpoint of the current domain
        """
        self._selection_strategy = selection_strategy
        self._selection_strategy_solver = selection_strategy_solver
        self._partition_strategy = partition_strategy


    def activate_verbose(self) -> None:
        """
        sets the values for logging.
        if no log is to be produced, will be False
        otw, a log will be produced

        Parameters 
        -----------
        verbose : bool
            if we want to produce a log or not
        """
        self._verbose = True


    def set_logging(self, 
                    log: bool = True, 
                    fname: str = "log",
                    level: str = "INFO") -> None:
        """
        sets the values for logging.
        if no log is to be produced, will be False
        otw, a log will be produced

        Parameters 
        -----------
        log : bool
            if we want to produce a log or not
        fname : str, optional
            specifies the name of the file the log will live in
        level : str, optional
            specifies if we just want general info (opt: INFO)
            or if we want comprehensive debugging data (opt: DEBUG)
        """
        assert type(log) == bool
        assert type(fname) == str
        assert type(level) == str
        assert level == "INFO" or level == "DEBUG"

        if log: 
            self._log = True
            self._logname = fname
            self._loglevel = level
        else: 
            self._log = False

        
    def inherit_solutions_from_parent(self,
                                      inherit: bool = True) -> None:
        """
        Because the LB problems are solved to global optimality,
        we should only have to solve a node if the bounds of the 
        new node overlap that of the original solution.

        Example:
            x is bounded between (0,1)
            y[s], where s \in S = [1,...,n]

            root node is solved, we have solutions x_root[s], y_root[s]
            & we spawn children into 
                - child1: x bounded by (0, 0.5)
                - child2: x bounded by (0.5, 1)
            
            say we select child1 and solve.
            for all s \in S:
                ** because we solved all of the root node LBs globally **
                - if x_root[s] is within the bounds of child1 = (0, 0.5)
                  then we can directly take the solution of the parent!
                - otw, we need to solve the model.
        
        This flag indicates if we would like to inherit from the parent solution,
        otherwise we will skip and always solve every subproblem.

        Parameters 
        -----------
        inherit : bool, optional
            indicates if we will inherit soultions of not
            default, True
        """
        assert type(inherit)==bool
        self._inhert_solutions = inherit
    

    def display(self):
        """
        Displays current settings.
        """
        print("-"*50)
        print("SNoGloDe Solver Parameters")
        print("-"*50)

        # display subproblems
        print("\nSubproblem Names:")
        for name in self._subproblem_names:
            print(f"  - {name}")
        print(f"Total # subproblems: {len(self._subproblem_names)}")
        
        # display solver information
        print(f"\nLB solver: {self._lb_solver.name}" )
        print(f"UB solver: {self._ub_solver.name}" )
        if self._cg_solver == None: print(f"CG solver: None")
        else: print(f"CG solver: {self._cg_solver.name}")

        # optional settings
        print(f"\nGlobal convergance guaranteed: {self._global_guarantee}")
        if len(self._rank_subproblem_names) > 0: 
            print("\nRank Subproblem Names:")
            for rank_name in self._rank_subproblem_names:
                print(f"  - {rank_name}")
        print(f"Epsilon: {self._epsilon}")
        print(f"Perform FBBT: {self._fbbt}")
        print(f"Perform OBBT: {self._obbt}")
        print(f"Relax binaries at LB: {self._relax_binaries}")
        print(f"Relax integers at LB: {self._relax_integers}")

        print("\nBounder Information")
        print(f"  - Lower Bounder: {self._lower_bounder.__name__}")
        print(f"  - Candidate Generator: {self._candidate_solution_finder.__name__}\n")

        print(f"Queuing strategy: {self._queue_strategy}")
        print(f"Using a node feasibility checker? : {self._node_feasibility_check != None}")
        
        print("\nBranching Information")
        print(f"  - Selection Strategy: {self._selection_strategy.__name__}")
        print(f"  - Partition Strategy: {self._partition_strategy.__name__}\n")

        print(f"Verbose: {self._verbose}")
        
        if self._log: print(f"Logging file saved to: {self._logname}")
        else: print("No log file generated.")
        print()
        print("-"*50)