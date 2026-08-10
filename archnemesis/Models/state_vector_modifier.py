
from typing import TYPE_CHECKING
import dataclasses as dc

import numpy as np

if TYPE_CHECKING:
    nx = 'number of elements in state vector'

@dc.dataclass(slots=True)
class StateVectorModifier:
    """
    Holds methods and attributes required to push and pull values from state vectors.
    """
    state_vector_start     : int = dc.field(default=-1, init=False)
    n_state_vector_entries : int = dc.field(default=-1, init=False)
    state_vector_slice     : slice = dc.field(default=dc.MISSING, init=False)
    
    def check(self):
        assert self.state_vector_start >= 0, "StateVectorModifier must have been initialised before any operations are performed"
        assert self.n_state_vector_entries >= 0, "StateVectorModifier must have been initialised before any operations are performed"
        assert isinstance(self.state_vector_slice, slice), "StateVectorModifier must have been initialised before any operations are performed"
    
    def set_state_vector_region(
            self, 
            idx_start : int,
            n : int,
    ):
        """
        Sets region of state vector to operate upon.
        """
        self.state_vector_start = idx_start
        self.n_state_vector_entries = n
        self.state_vector_slice = slice(self.state_vector_start, self.state_vector_start+self.n_state_vector_entries)
    
    def pull_from_state_vector(
            self,
            x0 : np.ndarray[["nx"], float],
            lx : None | np.ndarray[["nx"],int] = None,
    ):
        """
        Default method to update values from state vector
        """
        self.check()
        
        x0_part = x0[self.state_vector_slice]
        if lx is not None:
            lx_part = lx[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            stateparam.v = x0_part[stateparam.slice]
            if lx is not None:
                logv = lx_part[stateparam.slice]
                assert np.all(logv[0] == logv), "All log vector entries for a stateparam must be identical"
                assert stateparam.log == logv[0], "Log vector entries for a stateparam must agree with the stateparam's log value."
    
    def pull_from_covariance_matrix(
            self,
            sx : np.ndarray[["nx","nx"],float],
    ):
        """
        Default method to update errors from state vector
        """
        self.check()
        
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
    
        for stateparam in self.iter_stateparam_objs():
            if stateparam.log:
                stateparam.e = np.sqrt(np.diag(sx_part[stateparam.slice, stateparam.slice])*(stateparam.v**2))
            else:
                stateparam.e = np.sqrt(np.diag(sx_part[stateparam.slice, stateparam.slice]))
    
    def push_to_state_vector(
            self,
            x0 : np.ndarray[["nx"], float],
            lx : None | np.ndarray[["nx"],int] = None,
    ):
        """
        Default push to state vector
        """
        self.check()
        
        x0_part = x0[self.state_vector_slice]
        if lx is not None:
            lx_part = lx[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            print(f'{stateparam.v=}')
            x0_part[stateparam.slice] = stateparam.v
            if lx is not None:
                lx_part[stateparam.slice] = stateparam.log
    
    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["nx","nx"],float],
            sxminfac : float,
    ):
        """
        Default covariance matrix calculation only considers diagonal elements
        """
        self.check()
        
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
        sx_part[...] = 0.0
    
        for stateparam in self.iter_stateparam_objs():
            if stateparam.log:
                sx_part[stateparam.slice, stateparam.slice] = np.diag((stateparam.e / stateparam.v)**2)
            else:
                sx_part[stateparam.slice, stateparam.slice] = np.diag(stateparam.e**2)
    
    
