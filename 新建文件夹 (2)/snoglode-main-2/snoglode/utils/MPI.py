"""
Enables MPI to be used. 

If we do not have MPI available, then 
we create a fake COMM_WORLD to use in it's place.
"""

try:
    import mpi4py
    from mpi4py.MPI import *
    _haveMPI = True

except ImportError:
    print("No mpi4py detected - running in serial.")
    _haveMPI = False

    import numpy as _np
    import copy as _cp

    SUM = _np.sum
    PROD = _np.prod
    MAX = _np.max
    MIN = _np.min
    
    class _MockMPIComm:

        @property
        def rank(self):
            return 0
    
        @property
        def size(self):
            return 1
    
        def Get_rank(self):
            return self.rank
    
        def Get_size(self):
            return self.size
    
        def barrier(self):
            pass

        def Barrier(self):
            pass

        def allreduce(self, sendobj, op=SUM):
            return _cp.deepcopy(sendobj)

        def allreduce(self, sendobj, op=PROD):
            return _cp.deepcopy(sendobj)
        
        def allreduce(self, sendobj, op=MAX):
            return _cp.deepcopy(sendobj)
        
        def allreduce(self, sendobj, op=MIN):
            return _cp.deepcopy(sendobj)
    
        def bcast(self, data, root=0):
            return data
    
    COMM_WORLD = _MockMPIComm()