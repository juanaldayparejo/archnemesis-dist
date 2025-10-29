#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# Retrievals.py - Subroutines to perform atmospheric retrievals.
#
# Copyright (C) 2025 Juan Alday, Joseph Penn, Patrick Irwin,
# Jack Dobinson, Jon Mason, Jingxuan Yang
#
# This file is part of archNEMESIS.
#
# archNEMESIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations #  for 3.9 compatability
import archnemesis as ans
from archnemesis.enums import RetrievalStrategy
import time
import numpy as np

import logging
_lgr = logging.getLogger(__name__)


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
    #    USE INFORMATION FROM PREVIOUS RETRIEVALS 
    ######################################################
    ######################################################

    if Retrieval.LIN>0:

        #Reading .pre file
        Variables_prev = ans.Files.read_pre(runname)
        
        if Retrieval.LIN==1:
            
            _lgr.info('lin=1 :: Using information from previous retrieval to update parameters in reference classes and calculate the error from the previous covariance matrix in new measurement')
            
            #Calculating forward model
            FM_prev = ans.ForwardModel_0(Atmosphere=Atmosphere,Measurement=Measurement,Spectroscopy=Spectroscopy,Scatter=Scatter,Stellar=Stellar,Surface=Surface,CIA=CIA,Layer=Layer,Variables=Variables_prev,Telluric=Telluric)
            YN,KK = FM_prev.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
            
            #Calculating forward modelling error
            SF = KK @ Variables_prev.SA @ KK.T

            #Adding forward model error to the new measurement error
            Measurement.SE += SF

            #Updating reference classes with the ones from the forward model class, which have been updated using the previous retrievals
            #Note we do not update the measurement class as that was different in the previous retrieval
            Atmosphere = FM_prev.AtmosphereX
            Spectroscopy = FM_prev.SpectroscopyX
            Scatter = FM_prev.ScatterX
            Stellar = FM_prev.StellarX
            Surface = FM_prev.SurfaceX
            CIA = FM_prev.CIAX
            Layer = FM_prev.LayerX

        elif Retrieval.LIN==2:
            
            _lgr.info('lin=2 :: Using information from previous retrieval to update a priori covariance matrix')
            
            #Reading .pre file
            Variables_prev = ans.Files.read_pre(runname)
            
            ix1 = 0
            for ivar1 in range(Variables.NVAR):
                
                ix2 = 0
                for ivar2 in range(Variables_prev.NVAR):
                    
                    if np.all(Variables.VARIDENT[ivar1, :] == Variables_prev.VARIDENT[ivar2, :]):
                        _lgr.info('Updating variable '+str(Variables.VARIDENT[ivar1,:]))
                        
                        #update things
                        sa_prev = Variables_prev.SA[ix2:ix2+Variables_prev.NXVAR[ivar2],ix2:ix2+Variables_prev.NXVAR[ivar2]]
                        xn_prev = Variables_prev.XN[ix2:ix2+Variables_prev.NXVAR[ivar2]]
                        
                        Variables.SA[ix1:ix1+Variables.NXVAR[ivar1],ix1:ix1+Variables.NXVAR[ivar1]] = sa_prev
                        Variables.XN[ix1:ix1+Variables.NXVAR[ivar1]] = xn_prev
                        Variables.XA[ix1:ix1+Variables.NXVAR[ivar1]] = xn_prev

                    ix2 += Variables_prev.NXVAR[ivar2]
                ix1 += Variables.NXVAR[ivar1]
                    
        elif Retrieval.LIN==3:   

            _lgr.info('lin=3 :: Using information from previous retrieval to update a priori covariance matrix and calculate the new forward model error')
            
            #Reading .pre file
            Variables_prev = ans.Files.read_pre(runname)
            
            #First of all, we update the state vector of the new retrieval with the one from the previous retrieval
            ix1 = 0
            for ivar1 in range(Variables.NVAR):
                
                ix2 = 0
                for ivar2 in range(Variables_prev.NVAR):
                    
                    if np.all(Variables.VARIDENT[ivar1, :] == Variables_prev.VARIDENT[ivar2, :]):
                        _lgr.info('Updating variable '+str(Variables.VARIDENT[ivar1,:]))
                        
                        #update things
                        sa_prev = Variables_prev.SA[ix2:ix2+Variables_prev.NXVAR[ivar2],ix2:ix2+Variables_prev.NXVAR[ivar2]]
                        xn_prev = Variables_prev.XN[ix2:ix2+Variables_prev.NXVAR[ivar2]]
                        
                        Variables.SA[ix1:ix1+Variables.NXVAR[ivar1],ix1:ix1+Variables.NXVAR[ivar1]] = sa_prev
                        Variables.XN[ix1:ix1+Variables.NXVAR[ivar1]] = xn_prev
                        Variables.XA[ix1:ix1+Variables.NXVAR[ivar1]] = xn_prev

                    ix2 += Variables_prev.NXVAR[ivar2]
                ix1 += Variables.NXVAR[ivar1]

            #Now we calculate the forward model error from the previous retrieval
            FM_prev = ans.ForwardModel_0(Atmosphere=Atmosphere,Measurement=Measurement,Spectroscopy=Spectroscopy,Scatter=Scatter,Stellar=Stellar,Surface=Surface,CIA=CIA,Layer=Layer,Variables=Variables_prev,Telluric=Telluric)
            YN,KK = FM_prev.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
            
            #We do not want to include the forward model error from variables that are retrieved again now
            ix1 = 0
            for ivar1 in range(Variables.NVAR):
                
                ix2 = 0
                for ivar2 in range(Variables_prev.NVAR):
                    
                    if np.all(Variables.VARIDENT[ivar1, :] == Variables_prev.VARIDENT[ivar2, :]):
                        
                        _lgr.info('Removing forward model error contribution from variable '+str(Variables.VARIDENT[ivar1,:]))
                        
                        #Removing contribution from KK
                        KK[:,ix2:ix2+Variables_prev.NXVAR[ivar2]] = 0.0
            
                    ix2 += Variables_prev.NXVAR[ivar2]
                ix1 += Variables.NXVAR[ivar1]
            
            #Calculating forward modelling error
            SF = KK @ Variables_prev.SA @ KK.T

            #Adding forward model error to the new measurement error
            Measurement.SE += SF

            #Updating reference classes with the ones from the forward model class, which have been updated using the previous retrievals
            #Note we do not update the measurement class as that was different in the previous retrieval
            Atmosphere = FM_prev.AtmosphereX
            Spectroscopy = FM_prev.SpectroscopyX
            Scatter = FM_prev.ScatterX
            Stellar = FM_prev.StellarX
            Surface = FM_prev.SurfaceX
            CIA = FM_prev.CIAX
            Layer = FM_prev.LayerX

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
            Retrieval.write_raw(runname,Variables,Atmosphere)
            
    if retrieval_method == RetrievalStrategy.Nested_Sampling:
        Retrieval.make_plots()

    #Finishing pogram
    end = time.time()
    _lgr.info('Model run OK')
    _lgr.info(' Elapsed time (s) = '+str(end-start))
