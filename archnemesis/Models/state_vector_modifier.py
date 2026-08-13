
from typing import TYPE_CHECKING
import dataclasses as dc

import numpy as np

from .ModelParameterEntry import ModelParameterEntry

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

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
    
    def check_state_vector_region_valid(self):
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
    
    def get_stateparam_entries_from_vectors(
            self,
            apriori_state_vector : np.ndarray,
            posterior_state_vector : np.ndarray,
            apriori_covariance_matrix : np.ndarray,
            posterior_covariance_matrix : np.ndarray,
            lx : np.ndarray,
            fix : np.ndarray,
    ) -> dict[str,ModelParameterEntry,...]:
        asv_part = apriori_state_vector[self.state_vector_slice]
        psv_part = posterior_state_vector[self.state_vector_slice]
        acm_part = np.diag(apriori_covariance_matrix[self.state_vector_slice, self.state_vector_slice])
        pcm_part = np.diag(posterior_covariance_matrix[self.state_vector_slice, self.state_vector_slice])
        fx_part = fix[self.state_vector_slice]
        lx_part = lx[self.state_vector_slice]
        
        result = dict()
        
        for name, stateparam in self.iter_stateparam_items():
            lx_bit = lx_part[stateparam.slice]
            fx_bit = fx_part[stateparam.slice]
            asv_bit = asv_part[stateparam.slice]
            psv_bit = psv_part[stateparam.slice]
            acm_bit = acm_part[stateparam.slice]
            pcm_bit = pcm_part[stateparam.slice]
            
            asv_bit = np.where(lx_bit == 0, asv_bit, np.exp(asv_bit))
            psv_bit = np.where(lx_bit == 0, psv_bit, np.exp(psv_bit))
            
            acm_bit = np.where(lx_bit == 0, np.sqrt(acm_bit), np.sqrt(acm_bit*(asv_bit**2)))
            pcm_bit = np.where(lx_bit == 0, np.sqrt(pcm_bit), np.sqrt(pcm_bit*(psv_bit**2)))

            ix = self.state_vector_slice.start + stateparam.slice.start

            result[name] = ModelParameterEntry(
                self.id,
                name,
                slice(ix, ix + len(asv_bit)),
                fx_bit == 1,
                asv_bit,
                acm_bit,
                psv_bit,
                pcm_bit
            )
        return result
        
    
    def pull_from_all_state(
            self,
            x0 : np.ndarray,
            lx : np.ndarray,
            inum : np.ndarray,
            sx : np.ndarray,
    ):
        self.pull_from_state_vector(x0, lx)
        self.pull_from_numerical_differentiation_vector(inum)
        self.pull_from_covariance_matrix(sx)
    
    def pull_from_state_vector(
            self,
            x0 : np.ndarray[["nx"], float],
            lx : None | np.ndarray[["nx"],int] = None,
    ):
        """
        Default method to update values from state vector
        """
        self.check_state_vector_region_valid()
        
        x0_part = x0[self.state_vector_slice]
        if lx is not None:
            self.pull_from_log_vector(lx)
        
        for stateparam in self.iter_stateparam_objs():
            stateparam.v = np.exp(x0_part[stateparam.slice]) if stateparam.log else x0_part[stateparam.slice]
            
            if stateparam.slice.stop is not None and (stateparam.slice.stop - stateparam.slice.start) == 1:
                stateparam.v = stateparam.v[0]
    
    
    def pull_from_log_vector(
            self,
            lx : np.ndarray[["nx"],int],
    ):
        """
        Default method to update values from state vector
        """
        self.check_state_vector_region_valid()
        lx_part = lx[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            logv = lx_part[stateparam.slice]
            assert np.all(logv[0] == logv), "All log vector entries for a stateparam must be identical"
            stateparam.log = logv[0]
    
    def pull_from_numerical_differentiation_vector(
            self,
            inum : np.ndarray[['nx'], int],
    ):
        """
        Default pull from numerical differentiation vector
        """
        self.check_state_vector_region_valid()
        inum_part = inum[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            num_diffv = inum_part[stateparam.slice]
            assert np.all(num_diffv[0] == num_diffv), "All entries in numerical differentiation vector for a stateparam must be identical"
            stateparam.num_diff = num_diffv[0]
    
    def pull_from_covariance_matrix(
            self,
            sx : np.ndarray[["nx","nx"],float],
    ):
        """
        Default method to update errors from state vector
        """
        self.check_state_vector_region_valid()
        
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
    
        for stateparam in self.iter_stateparam_objs():
            if stateparam.log:
                stateparam.e = np.sqrt(np.diag(sx_part[stateparam.slice, stateparam.slice])*(stateparam.v**2))
            else:
                stateparam.e = np.sqrt(np.diag(sx_part[stateparam.slice, stateparam.slice]))
            
            if stateparam.slice.stop is not None and (stateparam.slice.stop - stateparam.slice.start) == 1:
                stateparam.e = stateparam.e[0]
    
    def push_to_state_vector(
            self,
            x0 : np.ndarray[["nx"], float],
            lx : None | np.ndarray[["nx"],int] = None,
    ):
        """
        Default push to state vector
        """
        self.check_state_vector_region_valid()
        
        x0_part = x0[self.state_vector_slice]
        if lx is not None:
            lx_part = lx[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            x0_part[stateparam.slice] = np.log(stateparam.v) if stateparam.log else stateparam.v
            if lx is not None:
                lx_part[stateparam.slice] = stateparam.log
    
    def push_to_numerical_differentiation_vector(
            self,
            inum : np.ndarray[['nx'], int],
    ):
        """
        Default push to numerical differentiation vector
        """
        self.check_state_vector_region_valid()
        inum_part = inum[self.state_vector_slice]
        
        for stateparam in self.iter_stateparam_objs():
            inum_part[stateparam.slice] = stateparam.num_diff
        
    
    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["nx","nx"],float],
            sxminfac : float,
    ):
        """
        Default covariance matrix calculation only considers diagonal elements
        """
        self.check_state_vector_region_valid()
        
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
        sx_part[...] = 0.0
    
        for stateparam in self.iter_stateparam_objs():
            if stateparam.log:
                sx_part[stateparam.slice, stateparam.slice] = np.diag((stateparam.e / stateparam.v)**2)
            else:
                sx_part[stateparam.slice, stateparam.slice] = np.diag(stateparam.e**2)
    
    
    def calc_correlation_distance_and_covariance_matrix(
            self,
            variance_vector : np.ndarray,
            dist : np.ndarray,
            correlation_length : float,
    ) -> np.ndarray:
        if dist.ndim == 1:
            one = np.ones((dist.size, dist.size))
            distance_matrix = (one * dist[None,:]) - (one * dist[:,None])
        else:
            distance_matrix = dist
        cor_dist_mat = np.exp(-1*np.abs(distance_matrix / correlation_length))
        return cor_dist_mat, np.sqrt(variance_vector[:,None] @ variance_vector[None,:])
    
    def push_to_covariance_matrix_with_correlation_length(
            self,
            sx : np.ndarray[["nx","nx"],float],
            sxminfac : float,
            correlation_lengths : float | tuple[float,...],
            distances : None | np.ndarray | tuple[None | np.ndarray] = None,
    ):
        self.check_state_vector_region_valid()
        
        if isinstance(correlation_lengths, (np.number, float, int)):
            assert self._n_stateparams == 1, "If being used, correlation length can only be a single number if there is only one stateparam, otherwise must have one correlation length for each stateparam."
            correlation_lengths = np.ones((1,),dtype=type(correlation_lengths))*correlation_lengths
        
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
        sx_part[...] = 0.0
    
        for i, stateparam in enumerate(self.iter_stateparam_objs()):
            if stateparam.log:
                variance_vector = (stateparam.e / stateparam.v)**2
            else:
                variance_vector = stateparam.e**2
            
            sx_part2 = sx_part[stateparam.slice, stateparam.slice]
            
            if correlation_lengths[i] == 0:
                sx_part2[...] = np.diag(variance_vector)
            else:
                if distances is None:
                    dist = stateparam.v
                elif isinstance(dist, np.ndarray):
                    if distances.ndim == 1:
                        dist = distances[stateparam.slice] 
                    else:
                        dist = distances[stateparam.slice, stateparam.slice]
                elif distances[i] is None:
                    dist = stateparam.v
                else:
                    dist = dist[i]
                
                cor_dist_mat, covariance_mat = self.calc_correlation_distance_and_covariance_matrix(variance_vector, dist, correlation_lengths[i])
                sxminfac_mask = cor_dist_mat >= sxminfac
                sx_part2[sxminfac_mask] = covariance_mat[sxminfac_mask]