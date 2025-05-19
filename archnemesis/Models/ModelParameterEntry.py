
import dataclasses as dc

import numpy as np


@dc.dataclass
class ModelParameterEntry:
    """
    A holder class that holds the value(s) of a model parameter.
    
    # ATTRIBUTES #
        
        model_id : int
            The id of the model that the state vector entry is associated with
        
        name : str
            Name of the parameter
        
        sv_slice : slice
            Slice of the state vector that the parameter is associated with. sv_slice.end - sv_slice.start = 'm'
            where 'm' is the number of entries in the state vector associated with the parameter.
        
        is_fixed : np.ndarray[['m'],bool]
            Array of boolean flags for each element of the parameter, if True the element is fixed (i.e. it is
            not retrieved and should be identical between the apriori/posterior values)
        
        apriori_value : np.ndarray[['m'],float]
            Value of each element of the parameter in the apriori state vector. The value has been 'unlogged'
            where applicable.
        
        posterior_value : np.ndarray[['m'],float]
            Value of each element of the parameter in the posterior state vector. The value has been 'unlogged'
            where applicable.
    """
    model_id : int # ID number of the model this is a parameter for
    sv_slice : slice # slice of state vector that contains the values for this parameter
    is_fixed : bool # flag to indicate if the parameter is fixed or can the apriori_value and posterior_value be different?
    apriori_value : np.ndarray[['m'],float] # value before retrievel
    posterior_value : np.ndarray[['m'],float] # value after retrieval