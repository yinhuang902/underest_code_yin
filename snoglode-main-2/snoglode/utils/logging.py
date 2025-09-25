"""
A log for a run will track all of the necessary
information to understand the progression of the snoglode code.

There are two levels of information - INFO and DEBUG.
Currently, only implemented timing and INFO
"""
import time
import numpy as np

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

class IterLogger():
    """
    For each iteration, there is a lot of info we want to save.
    This manages that data to eventually be dumped to a log and 
    rewritten PER ITERATION.
    """
    def __init__(self, 
                 log_name: str,
                 log_level: str):
        """
        We only need to indicate which level of logging we want.

        Parameters
        ----------
        log_name : str
            path name to the log file to write to 
        log_level : str
            Should either be INFO or DEBUG
        """
        assert type(log_level) == str
        assert log_level == "INFO" or log_level == "DEBUG", \
            "log_level should either be INFO or DEBUG."
        self.log_level = log_level

        assert type(log_name) == str
        self.logfile = open(log_name+".txt", "w")
        self.logfile.write("-"*15 + " SNoGloDe Log " + "-"*15 + "\n")
        
        # INFO level data
        self.iter = 0
        
        # [start time, stop time, total time]
        self.init = np.zeros(2)
        self.node_feasibility = np.zeros(2)
        self.lb = np.zeros(2)
        self.cg = np.zeros(2)
        self.ub = np.zeros(2)
        self.tree_bounding = np.zeros(2)
        self.tree_branching = np.zeros(2)

        # get the global start to help output times easier - only need this for INIT calculations
        MPI.COMM_WORLD.barrier()
        self.global_start = time.perf_counter()

        # for when we star timing the algorithm itself
        self.snoglode_start = None # to be updated

        # track total time
        self.total_init = 0
        self.total_node_feasibility = 0
        self.total_lb = 0
        self.total_cg = 0
        self.total_ub = 0
        self.total_tree_bounding = 0
        self.total_tree_branching = 0


    def alg_start(self, time):
        self.snoglode_start = time

    def alg_stop(self):
        self.snoglode_end = time.perf_counter()
        self.snoglode_total = self.snoglode_end - self.snoglode_start

    def init_start(self):
        self.init[0] = time.perf_counter()

    def init_stop(self):
        self.init[1] = time.perf_counter()
        self.total_init += self.init[1] - self.init[0]

    def node_feas_start(self):
        self.node_feasibility[0] = time.perf_counter()

    def node_feas_stop(self):
        self.node_feasibility[1] = time.perf_counter()
        self.total_node_feasibility += self.node_feasibility[1] - self.node_feasibility[0]
    
    def lb_start(self):
        self.lb[0] = time.perf_counter()

    def lb_stop(self):
        self.lb[1] = time.perf_counter()
        self.total_lb += self.lb[1] - self.lb[0]

    def cg_start(self):
        self.cg[0] = time.perf_counter()

    def cg_stop(self):
        self.cg[1] = time.perf_counter()
        self.total_cg += self.cg[1] - self.cg[0]
    
    def ub_start(self):
        self.ub[0] = time.perf_counter()

    def ub_stop(self):
        self.ub[1] = time.perf_counter()
        self.total_ub += self.ub[1] - self.ub[0]

    def bounding_start(self):
        self.tree_bounding[0] = time.perf_counter()

    def bounding_stop(self):
        self.tree_bounding[1] = time.perf_counter()
        self.total_tree_bounding += self.tree_bounding[1] - self.tree_bounding[0]
    
    def branching_start(self):
        self.tree_branching[0] = time.perf_counter()

    def branching_stop(self):
        self.tree_branching[1] = time.perf_counter()
        self.total_tree_branching += self.tree_branching[1] - self.tree_branching[0]

    def update(self):
        """
        Updates the log file for this iteration.
        Takes into consideration the level of logging necessary.
        """
        floating_point = 6
        MPI.COMM_WORLD.barrier()

        # only write init if it is iter = 0
        if self.iter == 0:
            init_start = round(MPI.COMM_WORLD.allreduce(self.init[0], op=MPI.MIN) - self.global_start, floating_point)
            init_stop = round(MPI.COMM_WORLD.allreduce(self.init[1], op=MPI.MAX) - self.global_start, floating_point)
            if (rank==0): self.logfile.write("init," + str(init_start) + "," + str(init_stop) + "\n")

        self.logfile.write("k," + str(self.iter) + "\n")

        lb_start = round(MPI.COMM_WORLD.allreduce(self.lb[0], op=MPI.MIN) - self.snoglode_start, floating_point)
        lb_end = round(MPI.COMM_WORLD.allreduce(self.lb[1], op=MPI.MAX) - self.snoglode_start, floating_point)
        if (rank==0): self.logfile.write("LB," + str(lb_start) + "," + str(lb_end) + "," + "\n")

        cg_start = round(MPI.COMM_WORLD.allreduce(self.cg[0], op=MPI.MIN) - self.snoglode_start, floating_point)
        cg_end = round(MPI.COMM_WORLD.allreduce(self.cg[1], op=MPI.MAX) - self.snoglode_start, floating_point)
        if (rank==0): self.logfile.write("CG," + str(cg_start) + "," + str(cg_end) + "," + "\n")
        
        ub_start = round(MPI.COMM_WORLD.allreduce(self.ub[0], op=MPI.MIN) - self.snoglode_start, floating_point)
        ub_end = round(MPI.COMM_WORLD.allreduce(self.ub[1], op=MPI.MAX) - self.snoglode_start, floating_point)
        if (rank==0): self.logfile.write("UB," + str(ub_start) + "," + str(ub_end) + "," + "\n")

        tree_bounding_start = round(MPI.COMM_WORLD.allreduce(self.tree_bounding[0], op=MPI.MIN) - self.snoglode_start, floating_point)
        tree_bounding_end = round(MPI.COMM_WORLD.allreduce(self.tree_bounding[1], op=MPI.MAX) - self.snoglode_start, floating_point)
        if (rank==0): self.logfile.write("Bound," + str(tree_bounding_start) + "," + str(tree_bounding_end) + "," + "\n")

        tree_branching_start = round(MPI.COMM_WORLD.allreduce(self.tree_branching[0], op=MPI.MIN) - self.snoglode_start, floating_point)
        tree_branching_end = round(MPI.COMM_WORLD.allreduce(self.tree_branching[1], op=MPI.MAX) - self.snoglode_start, floating_point)
        if (rank==0): self.logfile.write("Branch," + str(tree_branching_start) + "," + str(tree_branching_end) + "," + "\n")

        # reset
        self.iter += 1
        self.init.fill(0)
        self.node_feasibility.fill(0)
        self.lb.fill(0)
        self.cg.fill(0)
        self.ub.fill(0)
        self.tree_bounding.fill(0)
        self.tree_branching.fill(0)

    def complete(self):
        """
        Called once, when the algorithm is completed.
        """
        self.alg_stop()

        if (rank==0):
            self.logfile.write("\n" + "-"*15 + " SNoGloDe Summary " + "-"*15 + "\n")
            total = self.snoglode_end - self.global_start
            self.logfile.write("\nTotal time: " + str(round(total, 4)) + "s\n")
            self.logfile.write("\nTotal time spent initializing: " + str(round(self.total_init, 4)) + "s\n")
            self.logfile.write("Total time spent solving LB problems: " + str(round(self.total_lb, 4)) + "s\n")
            self.logfile.write("Total time spent solving CG problems: " + str(round(self.total_cg, 4)) + "s\n")
            self.logfile.write("Total time spent solving UB problems: " + str(round(self.total_ub, 4)) + "s\n")
            self.logfile.write("Total time spent bounding the tree: " + str(round(self.total_tree_bounding, 4)) + "s\n")
            self.logfile.write("Total time spent branching the tree: " + str(round(self.total_tree_branching, 4)) + "s\n")
            remaining_time = self.total_init + self.total_lb + self.total_cg + self.total_ub + self.total_tree_bounding + self.total_tree_branching
            self.logfile.write("Remaining time spent somewhere else in the code: " + str(round(total - remaining_time, 4)) + "s\n\n")

            self.logfile.write("Percent time spent initializing: " + str(round(self.total_init*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Percent time spent solving LB problems: " + str(round(self.total_lb*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Percent time spent solving CG problems: " + str(round(self.total_cg*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Percent time spent solving UB problems: " + str(round(self.total_ub*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Percent time spent bounding the tree: " + str(round(self.total_tree_bounding*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Percent time spent branching the tree: " + str(round(self.total_tree_branching*100/self.snoglode_total, 4)) + "%\n")
            self.logfile.write("Perfect time spent somewhere else in the code: " + str(round(((total - remaining_time)*100)/self.snoglode_total, 4)) + "%")


class MockIterLogger():
    """
    rather than littering the code with if statements, 
    create a dummy logger.
    """
    def __init__(self): pass
    def alg_start(self, time): pass
    def alg_stop(self): pass
    def init_start(self): pass
    def init_stop(self): pass
    def node_feas_start(self): pass
    def node_feas_stop(self): pass
    def lb_start(self): pass
    def lb_stop(self): pass
    def cg_start(self): pass
    def cg_stop(self): pass
    def ub_start(self): pass
    def ub_stop(self): pass
    def bounding_start(self): pass
    def bounding_stop(self): pass
    def branching_start(self): pass
    def branching_stop(self): pass
    def update(self): pass
    def complete(self): pass


class LogWalker():
    def __init__(self):
        pass