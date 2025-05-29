from __future__ import annotations #  for 3.9 compatability
import archnemesis as ans
from archnemesis.enums import RetrievalStrategy
import numpy as np
import matplotlib.pyplot as plt
import time


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

def retrieval_nemesis(
        runname,
        legacy_files=False,
        NCores=1,
        retrieval_method : RetrievalStrategy = RetrievalStrategy.Optimal_Estimation,
        nemesisSO=False,
        NS_prefix='chains/'
    ):
    
    """
        FUNCTION NAME : retrieval_nemesis()
        
        DESCRIPTION :
        
            Function to run a NEMESIS retrieval based on the information in the input files
        
        INPUTS :
        
            runname :: Name of the retrieval run (i.e., name of the input files)
        
        OPTIONAL INPUTS:

            legacy_files :: If True, it reads the inputs from the standard Fortran NEMESIS files
                            If False, it reads the inputs from the archNEMESIS HDF5 file
            NCores :: Number of parallel processes for the numerical calculation of the Jacobian
            retrieval_method :: (0) Optimal Estimation formalism
                                (1) Nested sampling
            nemesisSO :: If True, it indicates that the retrieval is a solar occultation observation
        
        OUTPUTS :
        
            Output files
        
        CALLING SEQUENCE:
        
            retrieval_nemesis(runname,legacy_files=False,NCores=1)
        
        MODIFICATION HISTORY : Juan Alday (21/09/2024)
        
    """ 
    
    start = time.time()

    ######################################################
    ######################################################
    #    READING INPUT FILES AND SETTING UP VARIABLES
    ######################################################
    ######################################################

    if legacy_files is False:
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval,Telluric = ans.Files.read_input_files_hdf5(runname)
    else:
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
        Telluric = None

    ######################################################
    ######################################################
    #      RUN THE RETRIEVAL USING ANY APPROACH
    ######################################################
    ######################################################

    if retrieval_method == RetrievalStrategy.Optimal_Estimation:
        OptimalEstimation = ans.coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric,\
                                          NITER=Retrieval.NITER,PHILIMIT=Retrieval.PHILIMIT,NCores=NCores,nemesisSO=nemesisSO)
        Retrieval = OptimalEstimation
    elif retrieval_method == RetrievalStrategy.Nested_Sampling:
        from archnemesis.NestedSampling_0 import coreretNS
        
        NestedSampling = coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric,NS_prefix=NS_prefix)
        Retrieval = NestedSampling
    else:
        raise ValueError('error in retrieval_nemesis :: Retrieval scheme has not been implemented yet')


    ######################################################
    ######################################################
    #                WRITE OUTPUT FILES
    ######################################################
    ######################################################

    if retrieval_method == RetrievalStrategy.Optimal_Estimation:
        
        if legacy_files is False:
            Retrieval.write_output_hdf5(runname,Variables)
        else:
            Retrieval.write_cov(runname,Variables,pickle=False)
            Retrieval.write_mre(runname,Variables,Measurement)
            
    if retrieval_method == RetrievalStrategy.Nested_Sampling:
        Retrieval.make_plots()

    #Finishing pogram
    end = time.time()
    _lgr.info('Model run OK')
    _lgr.info(' Elapsed time (s) = '+str(end-start))
