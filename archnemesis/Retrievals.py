import archnemesis as ans
import numpy as np
import matplotlib.pyplot as plt
import time


def retrieval_nemesis(runname,legacy_files=False,NCores=1,retrieval_method=0,nemesisSO=False):
    
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
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files_hdf5(runname)
    else:
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)

    ######################################################
    ######################################################
    #      RUN THE RETRIEVAL USING ANY APPROACH
    ######################################################
    ######################################################

    if retrieval_method==0:
        OptimalEstimation = ans.coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,\
                                          NITER=Retrieval.NITER,PHILIMIT=Retrieval.PHILIMIT,NCores=NCores,nemesisSO=nemesisSO)
        Retrieval = OptimalEstimation
    else:
        sys.exit('error in retrieval_nemesis :: Retrieval scheme has not been implemented yet')


    ######################################################
    ######################################################
    #                WRITE OUTPUT FILES
    ######################################################
    ######################################################

    if retrieval_method==0:
        
        if legacy_files is False:
            Retrieval.write_output_hdf5(runname,Variables)
        else:
            Retrieval.write_cov(runname,Variables,pickle=False)
            Retrieval.write_mre(runname,Variables,Measurement)

    #Finishing pogram
    end = time.time()
    print('Model run OK')
    print(' Elapsed time (s) = '+str(end-start))