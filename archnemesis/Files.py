#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# Files.py - Functions to read and write the input/output files for an archNEMESIS simulation.
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



import os
import textwrap

import numpy as np
import matplotlib.pyplot as plt

import archnemesis as ans
from archnemesis import (
    Atmosphere_0,
    Variables_0,
    Measurement_0,
    Layer_0,
    OptimalEstimation_0,
    CIA_0,
    Scatter_0
)
from archnemesis.Layer_0 import read_play,read_hlay
from archnemesis.Data.planet_data import planet_info

from archnemesis.Models import Models, ModelBase, ModelParameterEntry
from copy import copy

from archnemesis.helpers import h5py_helper, io_helper
from archnemesis.enums import (
    #PlanetEnum, 
    #AtmosphericProfileFormatEnum, 
    #InstrumentLineshape, 
    LayerType,
    WaveUnit, 
    SpectraUnit,
    SpectralCalculationMode, 
    LowerBoundaryCondition, 
    ScatteringCalculationMode, 
    AerosolPhaseFunctionCalculationMode,
    ParaH2Ratio, 
    RayleighScatteringMode,
    LayerIntegrationScheme,
)

import logging
_lgr = logging.getLogger(__name__)

###############################################################################################
###############################################################################################
#                                            GENERIC
###############################################################################################
###############################################################################################

########################################################################################

def file_lines(fname):

    """
    FUNCTION NAME : file_lines()
    
    DESCRIPTION : Returns the number of lines in a given file
    
    INPUTS : 
 
        fname :: Name of the file

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        nlines :: Number of lines in file

    CALLING SEQUENCE:

        nlines = file_lines(fname)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


###############################################################################################
###############################################################################################
#                                   archNEMESIS FILES
###############################################################################################
###############################################################################################

# read_input_files_hdf5()
# read_retparam_hdf5()
# read_bestfit_hdf5()

###############################################################################################

def read_input_files_hdf5(runname,calc_SE=True):

    """
        FUNCTION NAME : read_input_files_hdf5()
        
        DESCRIPTION : 

            Reads the NEMESIS HDF5 input file and fills the parameters in the reference classes.
 
        INPUTS :
      
            runname :: Name of the NEMESIS run

        OPTIONAL INPUTS:
        
            calc_SE :: If True, it will calculate the Measurement error matrix SE (useful only for retrievals, not for forward models)
        
        OUTPUTS : 

            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations

        CALLING SEQUENCE:
        
            Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval,Telluric = read_input_files_hdf5(runname)
 
        MODIFICATION HISTORY : Juan Alday (25/03/2023)
    """

    from archnemesis import OptimalEstimation_0,Layer_0,Surface_0,Scatter_0,CIA_0,Measurement_0,Spectroscopy_0,Stellar_0, Telluric_0
    import h5py

    #Initialise Atmosphere class and read file
    ##############################################################

    Atmosphere = Atmosphere_0(runname=runname)

    #Read gaseous atmosphere
    Atmosphere.read_hdf5(runname)
    
    #Initialise Layer class and read file
    ###############################################################

    Layer = Layer_0(Atmosphere.RADIUS)
    Layer.read_hdf5(runname)

    #Initialise Surface class and read file
    ###############################################################

    #isurf = planet_info[str(int(Atmosphere.IPLANET))]["isurf"]
    Surface = Surface_0()
    # Always read the Surface entry in HDF5 file as defaults are set
    # there for gas giants that are different for the defaults
    # when constructing via Surface_0()
    Surface.read_hdf5(runname)
    if np.mean(Surface.TSURF)<=0.0:
        Surface.GASGIANT=True   #If T is negative then we omit the surface


    #Initialise Scatter class and read file
    ###############################################################

    Scatter = Scatter_0()
    Scatter.read_hdf5(runname)

    #Initialise CIA class and read files (.cia)  - NOT FROM HDF5 YET
    ##############################################################

    with h5py.File(runname+'.h5','r') as f:
        #Checking if CIA exists
        e = "/CIA" in f
    
    if e==True:
        CIA = CIA_0(runname=runname)
        CIA.read_hdf5(runname)
    else:
        CIA = None

    #Old version of CIA
    #if os.path.exists(runname+'.cia')==True:
    #    CIA = CIA_0(runname=runname)
    #    CIA.read_cia()
    #    #CIA.read_hdf5(runname)
    #else:
    #    CIA = None

    #Initialise Measurement class and read file
    ###############################################################

    Measurement = Measurement_0(runname=runname)
    Measurement.read_hdf5(runname,calc_MeasurementVector=calc_SE)
    
    #Initialise Spectroscopy class and read file
    ###############################################################

    with h5py.File(runname+'.h5','r') as f:
        #Checking if Spectroscopy exists
        e = "/Spectroscopy" in f

    if e is True:
        Spectroscopy = Spectroscopy_0(RUNNAME=runname)
        Spectroscopy.read_hdf5(runname)
    else:
        Spectroscopy = None

    #Initialise Telluric class and read file
    ###############################################################
    
    with h5py.File(runname+'.h5','r') as f:
        #Checking if Telluric exists
        e = "/Telluric" in f
    
    if e is True:
        Telluric = Telluric_0()
        Telluric.read_hdf5(runname)
    else:
        Telluric = None

    #Reading Stellar class
    ################################################################

    Stellar = Stellar_0()
    Stellar.read_hdf5(runname)

    #Reading .apr file and Variables Class
    #################################################################

    Variables = Variables_0()
    Variables.read_apr(runname, Atmosphere.NP, Atmosphere.NVMR, Atmosphere.NDUST, Atmosphere.NLOCATIONS)
    Variables.XN = copy(Variables.XA)
    Variables.SX = copy(Variables.SA)

    #Reading retrieval setup
    #################################################################

    Retrieval = OptimalEstimation_0()
    Retrieval.read_hdf5(runname)

    return Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval,Telluric

###############################################################################################

def read_retparam_hdf5(runname):
    """

        DESCRIPTION : 

            Read the retrieved parameters from the HDF5 file

        INPUTS :
      
            runname :: Name of the NEMESIS run

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            nvar :: Number of retrieved model parameterisations
            nxvar(nvar) :: Number of parameters associated with each model parameterisation
            varident(nvar,3) :: Variable parameterisation ID
            varparam(nvar,nparam) :: Extra parameters required to model the parameterisations (not retrieved)
            aprparam(nxvar,nvar) :: A priori parameters required to model the parameterisations
            aprerrparam(nxvar,nvar) :: Uncertainty in the a priori parameters required to model the parameterisations 
            retparam(nxvar,nvar) :: Retrieved parameters required to model the parameterisations
            reterrparam(nxvar,nvar) :: Uncertainty in the retrieved parameters required to model the parameterisations


        CALLING SEQUENCE:
        
            nvar,nxvar,varident,varparam,aprparam,aprerrparam,retparam,reterrparam = read_retparam_hdf5(runname)
 
        MODIFICATION HISTORY : Juan Alday (25/03/2023)
    """

    import h5py

    with h5py.File(runname+'.h5','r') as f:

        #Checking if Retrieval exists
        if "/Retrieval" not in f:
            raise ValueError('error :: Retrieval is not defined in HDF5 file')
            return None
        
        if '/Retrieval/Output/Parameters' not in f:
            raise ValueError('error :: Retrieval/Output/Parameters is not defined in HDF5 file')
            return None

        NVAR = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/NVAR', np.int32)
        NXVAR = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/NXVAR', np.array)
        VARIDENT = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/VARIDENT', np.array)
        VARPARAM = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/VARPARAM', np.array)
        RETPARAM = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/RETPARAM', np.array)
        RETERRPARAM = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/RETERRPARAM', np.array)
        APRPARAM = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/APRPARAM', np.array)
        APRERRPARAM = h5py_helper.retrieve_data(f, 'Retrieval/Output/Parameters/APRERRPARAM', np.array)


    return NVAR,NXVAR,VARIDENT,VARPARAM,APRPARAM,APRERRPARAM,RETPARAM,RETERRPARAM

###############################################################################################

def read_bestfit_hdf5(runname):
    """

        DESCRIPTION : 

            Read the best fit from the HDF5 file and include it in the Measurement class

        INPUTS :
      
            runname :: Name of the NEMESIS run

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            Measurement :: Python class defining the measurement and best fit to the data

        CALLING SEQUENCE:
        
            Measurement = read_bestfit_hdf5(runname)
 
        MODIFICATION HISTORY : Juan Alday (25/03/2023)
    """

    import h5py

    #Reading the best fit
    with h5py.File(runname+'.h5','r') as f:
        if "/Retrieval" not in f:
            raise ValueError('error :: Retrieval is not defined in HDF5 file')
            return None
        
        if '/Retrieval/Output/OptimalEstimation/YN' not in f:
            raise ValueError('error :: Retrieval/Output/OptimalEstimation/YN is not defined in HDF5 file')
            return None
        
        YN = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/YN', np.array)
 
    #Writing the measurement vector in same format as in Measurement
    Measurement = Measurement_0()
    Measurement.read_hdf5(runname)

    SPECMOD = np.zeros(Measurement.MEAS.shape)
    ix = 0
    for i in range(Measurement.NGEOM):
        SPECMOD[0:Measurement.NCONV[i],i] = YN[ix:ix+Measurement.NCONV[i]]
        ix = ix + Measurement.NCONV[i]

    Measurement.edit_SPECMOD(SPECMOD)

    return Measurement

###############################################################################################
###############################################################################################
#                                     NEMESIS FILES
###############################################################################################
###############################################################################################

# read_input_files()

# read_mre()
# read_cov()
# read_drv()
# read_inp()
# read_set()
# read_fla()
# write_fla()
# write_set()
# write_inp()
# write_err()
# write_fcloud()


###############################################################################################

def read_input_files(runname):

    """
        FUNCTION NAME : read_input_files()
        
        DESCRIPTION : 

            Reads the NEMESIS input files and fills the parameters in the reference classes.
 
        INPUTS :
      
            runname :: Name of the NEMESIS run

        OPTIONAL INPUTS: None
        
        OUTPUTS : 

            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations
            Retrieval :: Python class defining the initial Optimal Esimation setup for the input data

        CALLING SEQUENCE:
        
            Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = read_input_files(runname)
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
    """

    #Initialise Atmosphere class and read file (.ref, aerosol.ref)
    ##############################################################

    _lgr.info('Reading atmospheric files...')
    Atm = Atmosphere_0(runname=runname)

    #Read gaseous atmosphere
    Atm.read_ref()

    #Read aerosol profiles
    Atm.read_aerosol()
    
    #Read para-h2
    Atm.read_parah2()
    
    #Read .vpf
    Atm.read_vpf()

    #Reading .set file and starting Scatter, Stellar, Surface and Layer Classes
    #############################################################################

    Layer = Layer_0(Atm.RADIUS)
    Scatter,Stellar,Surface,Layer = read_set(runname,Layer=Layer)
    if Layer.LAYTYP==LayerType.BASE_PRESSURE:
        nlay, pbase = read_play()
        Layer.NLAY = nlay
        Layer.P_base = pbase*101325 
    if Layer.LAYTYP==LayerType.BASE_HEIGHT:
        nlay,hbase = read_hlay()
        Layer.NLAY = nlay
        Layer.H_base = hbase*1.0e3    #Base height of each layer (m)
    if Layer.LAYTYP not in (LayerType.EQUAL_PRESSURE, LayerType.EQUAL_LOG_PRESSURE, LayerType.EQUAL_HEIGHT, LayerType.EQUAL_PATH_LENGTH, LayerType.BASE_PRESSURE, LayerType.BASE_HEIGHT):
        raise ValueError('error in read_input_files :: Need to read the press.lay file but not implemented yet')
    
    Layer.DUST_UNITS_FLAG = Atm.DUST_UNITS_FLAG

    #Reading .inp file and starting Measurement,Scatter and Spectroscopy classes
    #############################################################################

    Measurement,Scatter,Spec,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN = read_inp(runname,Scatter=Scatter)

    Measurement.WOFF = WOFF
    
    Retrieval = OptimalEstimation_0()
    Retrieval.NITER=NITER
    Retrieval.PHILIMIT=PHILIMIT
    Retrieval.LIN = LIN

    #Reading surface files if planet has surface
    #############################################################################

    isurf = planet_info[str(int(Atm.IPLANET))]["isurf"]
    if isurf==True:
        if np.mean(Surface.TSURF)>0.0:
            Surface.GASGIANT=False
            Surface.read_sur(runname) #Emissivity (and albedo for Lambert surface)
            if Surface.LOWBC==LowerBoundaryCondition.HAPKE: #Hapke surface
                Surface.read_hap(runname)
        else:
            Surface.GASGIANT=True
    else:
        Surface.GASGIANT=True
    
    if Surface.GASGIANT:
        Surface.LOWBC = LowerBoundaryCondition.THERMAL
        Surface.TSURF = 0.0
        Surface.GALB = 0.0

    #Reading Spectroscopy parameters from .lls or .kls files
    ##############################################################
    if Spec.ILBL==SpectralCalculationMode.K_TABLES:
        Spec.read_kls(runname)
    elif Spec.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:
        Spec.read_lls(runname)
    else:
        raise ValueError('error :: ILBL has to be either SpectralCalculationMode.K_TABLES or SpectralCalculationMode.LINE_BY_LINE_TABLES')

    #Reading extinction and scattering cross sections
    #############################################################################

    Scatter.read_xsc(runname)

    if Scatter.NDUST!=Atm.NDUST:
        raise ValueError('error :: Number of aerosol populations must be the same in .xsc and aerosol.ref files')


    #Initialise Measurement class and read files (.spx, .sha)
    ##############################################################

    Measurement.runname = runname
    Measurement.read_spx()
    
    #Checking if forward modelling error file exists
    if os.path.exists(fmerrname)==True:

        #Reading forward modelling error
        vfmerr,fmerr = read_fmerr(fmerrname)
        Measurement.VFMERR = vfmerr
        Measurement.FMERR = fmerr
        Measurement.add_fmerr()

    #Reading .sha file if FWHM>0.0
    if Measurement.FWHM > 0.0:
        Measurement.read_sha()
    #Reading .fil if FWHM<0.0
    elif Measurement.FWHM < 0.0:
        Measurement.read_fil()

    #Reading stellar spectrum if required by Measurement units
    if Measurement.IFORM in (SpectraUnit.FluxRatio, SpectraUnit.TransitDepth, SpectraUnit.Integrated_spectral_power, SpectraUnit.Atmospheric_transmission):
        Stellar.read_sol(runname)

    #Initialise CIA class and read files (.cia)
    ##############################################################

    if os.path.exists(runname+'.cia')==True:
        CIA = CIA_0(runname=runname)
        CIA.read_cia()
    else:
        CIA = None

    #Reading .fla file
    #############################################################################

    inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv = read_fla(runname)

    if CIA is not None:
        CIA.INORMAL = ParaH2Ratio(inormal)

    Scatter.IRAY = RayleighScatteringMode(iray)
    Scatter.IMIE = AerosolPhaseFunctionCalculationMode(imie)

    if Scatter.ISCAT!=ScatteringCalculationMode.THERMAL_EMISSION:
        if Scatter.IMIE==AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:
            Scatter.read_hgphase()
        elif Scatter.IMIE==AerosolPhaseFunctionCalculationMode.MIE_THEORY:
            Scatter.read_phase()
        elif Scatter.IMIE==AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:
            Scatter.read_lpphase()
        else:
            raise ValueError('error :: IMIE must be an integer from 0 to 2')

    #Reading .apr file and Variables Class
    #################################################################
    _lgr.info('Reading .apr file')
    Variables = Variables_0()
    Variables.read_apr(runname, Atm.NP, Atm.NVMR, Atm.NDUST, Atm.NLOCATIONS)
    Variables.XN = copy(Variables.XA)
    Variables.SX = copy(Variables.SA)

    return Atm,Measurement,Spec,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval

###############################################################################################

def read_mre(runname,MakePlot=False):

    """
    FUNCTION NAME : read_mre()

    DESCRIPTION : Reads the .mre file from a Nemesis run

    INPUTS :
    
        runname :: Name of the Nemesis run

    OPTIONAL INPUTS:
    
        MakePlot : If True, a summary plot is made
            
    OUTPUTS : 

        lat :: Latitude (degrees)
        lon :: Longitude (degrees)
        ngeom :: Number of geometries in the observation
        nconv :: Number of points in the measurement vector for each geometry (assuming they all have the same number of points)
        wave(nconv,ngeom) :: Wavelength/wavenumber of each point in the measurement vector
        specret(nconv,ngeom) :: Retrieved spectrum for each of the geometries
        specmeas(nconv,ngeom) :: Measured spectrum for each of the geometries
        specerrmeas(nconv,ngeom) :: Error in the measured spectrum for each of the geometries
        nx :: Number of points in the state vector
        varident(nvar,3) :: Retrieved variable ID, as defined in Nemesis manual
        nxvar :: Number of points in the state vector associated with each retrieved variable
        varparam(nvar,5) :: Extra parameters containing information about how to read the retrieved variables
        aprprof(nx,nvar) :: A priori profile for each variable in the state vector
        aprerr(nx,nvar) :: Error in the a priori profile for each variable in the state vector
        retprof(nx,nvar) :: Retrieved profile for each variable in the state vector
        reterr(nx,nvar) :: Error in the retrieved profile for each variable in the state vector

    CALLING SEQUENCE:

        lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr = read_mre(runname)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)

    """

    #Opening file
    f = open(runname+'.mre','r')

    #Reading first three lines
    tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
    s = f.readline().split()
    #nspec = int(tmp[0])
    tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
    s = f.readline().split()
    #ispec = int(tmp[0])
    ngeom = int(tmp[1])
    ny2 = int(tmp[2])
    ny = int(ny2 / ngeom)
    nx = int(tmp[3])
    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
    s = f.readline().split()
    lat = float(tmp[0])
    lon = float(tmp[1])
    
    #Reading spectra
    s = f.readline().split()
    s = f.readline().split()
    wave = np.zeros([ny,ngeom])
    specret = np.zeros([ny,ngeom])
    specmeas = np.zeros([ny,ngeom])
    specerrmeas = np.zeros([ny,ngeom])
    for i in range(ngeom):
        for j in range(ny):
            tmp = np.fromfile(f,sep=' ',count=7,dtype='float')
            wave[j,i] = float(tmp[1])
            specret[j,i] = float(tmp[5])
            specmeas[j,i] = float(tmp[2])
            specerrmeas[j,i] = float(tmp[3])

    #Reading the retrieved state vector
    s = f.readline().split()
    if len(s)==2:
        nvar = int(s[1])
    else:
        nvar = int(s[2])

    nxvar = np.zeros([nvar],dtype='int')
    
    Var = Variables_0()
    Var.NVAR = nvar
    aprprof1 = np.zeros([nx,nvar])
    aprerr1 = np.zeros([nx,nvar])
    retprof1 = np.zeros([nx,nvar])
    reterr1 = np.zeros([nx,nvar])
    varident = np.zeros([nvar,3],dtype='int')
    varparam = np.zeros([nvar,5])
    
    for i in range(nvar):
        
        # 1) Read until we find the line that starts with "Variable"
        line = f.readline()
        while line and not line.strip().startswith('Variable'):
            line = f.readline()
        if not line:
            # End of file or no more "Variable" lines
            break

        # 2) The next line is the varident (3 integers)
        line = f.readline().strip()
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Expected 3 integers for varident, got: {parts}")
        varident[i, :] = np.array(parts, dtype=int)

        # 3) The next line is the varparam (5 floats)
        line = f.readline().strip()
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 floats for varparam, got: {parts}")
        varparam[i, :] = np.array(parts, dtype=float)
            
        # 4) The next line is typically the header for data lines ("i, ix, xa ...")
        #    We just read it and ignore.
        f.readline()

        # 5) Now read each profile line for this variable
        #    We know the data lines each have 6 columns.
        varlines = []
        while True:
            pos = f.tell()  # Remember position before reading
            line = f.readline()
            if not line:
                break
            if line.strip().startswith('Variable'):
                f.seek(pos)  # Rewind so this line will be read again later
                break
            varlines.append(line)
        nxvar[i] = len(varlines)        

        for j in range(len(varlines)):
            line = varlines[j]
            parts = line.split()
            # If something is off, backtrack and break
            if len(parts) != 6:
                raise ValueError('error in read_mre :: something is off when reading the retrieved parameters')
            aprprof1[j, i] = float(parts[2])
            aprerr1[j, i]  = float(parts[3])
            retprof1[j, i] = float(parts[4])
            reterr1[j, i]  = float(parts[5])

    Var.edit_VARIDENT(varident)
    Var.edit_VARPARAM(varparam)
    Var.NXVAR = nxvar

    aprprof = np.zeros([Var.NXVAR.max(),nvar])
    aprerr = np.zeros([Var.NXVAR.max(),nvar])
    retprof = np.zeros([Var.NXVAR.max(),nvar])
    reterr = np.zeros([Var.NXVAR.max(),nvar])

    for i in range(Var.NVAR):
        aprprof[0:Var.NXVAR[i],i] = aprprof1[0:Var.NXVAR[i],i]
        aprerr[0:Var.NXVAR[i],i] = aprerr1[0:Var.NXVAR[i],i]
        retprof[0:Var.NXVAR[i],i] = retprof1[0:Var.NXVAR[i],i]
        reterr[0:Var.NXVAR[i],i] = reterr1[0:Var.NXVAR[i],i]

    return lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr

###############################################################################################

def read_cov(runname,MakePlot=False):
    
    
    """
        FUNCTION NAME : read_cov()
        
        DESCRIPTION :
        
            Reads the the .cov file with the standard Nemesis format
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            npro :: Number of points in atmospheric profiles
            nvar :: Number of retrieved variables
            varident(nvar,3) :: Variable ID
            varparam(nvar,mparam) :: Extra parameters for describing the retrieved variable
            nx :: Number of elements in state vector
            ny :: Number of elements in measurement vector
            sa(nx,nx) :: A priori covariance matric
            sm(nx,nx) :: Final measurement covariance matrix
            sn(nx,nx) :: Final smoothing error covariance matrix
            st(nx,nx) :: Final full covariance matrix
            se(ny,ny) :: Measurement error covariance matrix
            aa(nx,nx) :: Averaging kernels
            dd(nx,ny) :: Gain matrix
            kk(ny,nx) :: Jacobian matrix
        
        CALLING SEQUENCE:
        
            npro,nvar,varident,varparam,nx,ny,sa,sm,sn,st,se,aa,dd,kk = read_cov(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Open file
    f = open(runname+'.cov','r')
    
    #Reading variables that were retrieved
    tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
    npro = int(tmp[0])
    nvar = int(tmp[1])
    
    varident = np.zeros([nvar,3],dtype='int')
    varparam = np.zeros([nvar,5],dtype='int')
    for i in range(nvar):
        tmp = np.fromfile(f,sep=' ',count=3,dtype='int')
        varident[i,:] = tmp[:]
        
        tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
        varparam[i,:] = tmp[:]
    
    
    #Reading optimal estimation matrices
    tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
    nx = int(tmp[0])
    ny = int(tmp[1])


    sa = np.zeros([nx,nx])
    sm = np.zeros([nx,nx])
    sn = np.zeros([nx,nx])
    st = np.zeros([nx,nx])
    aa = np.zeros([nx,nx])
    dd = np.zeros([nx,ny])
    kk = np.zeros([ny,nx])
    se = np.zeros([ny,ny])
    for i in range(nx):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sa[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sm[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sn[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            st[i,j] = tmp[0]

    for i in range(nx):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            aa[i,j] = tmp[0]
    
    
    for i in range(nx):
        for j in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            dd[i,j] = tmp[0]

    for i in range(ny):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            kk[i,j] = tmp[0]
    
    for i in range(ny):
        tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
        se[i,i] = tmp[0]

    f.close()

    return npro,nvar,varident,varparam,nx,ny,sa,sm,sn,st,se,aa,dd,kk

###############################################################################################

def read_drv(runname,MakePlot=False):
    
    """
        FUNCTION NAME : read_drv()
        
        DESCRIPTION : Read the .drv file, which contains all the required information for
                      calculating the observation paths
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS:
        
            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            iconv :: Spectral model code
            flagh2p :: Flag for para-H2
            ncont :: Number of aerosol populations
            flagc :: Flag for
            nlayer :: Number of atmospheric layers
            npath :: Number of observation paths
            ngas :: Number of gases in atmosphere
            gasID(ngas) :: RADTRAN gas ID
            isoID(ngas) :: RADTRAN isotopologue ID (0 for all isotopes)
            iproc(ngas) :: Process parameter
            baseH(nlayer) :: Altitude of the base of each layer (km)
            delH(nlayer) :: Altitude covered by the layer (km)
            baseP(nlayer) :: Pressure at the base of each layer (atm)
            baseT(nlayer) :: Temperature at the base of each layer (K)
            totam(nlayer) :: Vertical total column density in atmosphere (cm-2)
            press(nlayer) :: Effective pressure of each layer (atm)
            temp(nlayer) :: Effective temperature of each layer (K)
            doppler(nlayer) ::
            par_coldens(nlayer,ngas) :: Vertical total column density for each gas in atmosphere (cm-2)
            par_press(nlayer,ngas) :: Partial pressure of each gas (atm)
            cont_coldens(nlayer,ncont) :: Aerosol column density for each aerosol population in atmosphere (particles per gram of atm)
            hfp(nlayer) ::
            hfc(nlayer,ncont) ::
            nlayin(npath) :: Number of layers seen in each path
            imod(npath) :: Path model
            errlim(npath) ::
            layinc(npath,2*nlayer) :: Layer indices seen in each path
            emtemp(npath,2*nlayer) :: Emission temperature of each layer in path
            scale(npath,2*nlayer) :: Factor to be applied to the vertical column density to calculate the line-of-sight column density
            nfilt :: Number of profile filter points
            filt(nfilt) :: Filter points
            vfilt(nfilt) ::
            ncalc :: Number of calculations
            itype(ncalc) :: Calculation type
            nintp(ncalc) ::
            nrealp(ncalc) ::
            nchp(ncalc) ::
            icald(ncalc,10) ::
            rcald(ncalc,10) ::
            
        CALLING SEQUENCE:
            
            iconv,flagh2p,ncont,flagc,nlayer,npath,ngas,gasID,isoID,iproc,\
            baseH,delH,baseP,baseT,totam,press,temp,doppler,par_coldens,par_press,cont_coldens,hfp,hfc,\
            nlayin,imod,errlim,layinc,emtemp,scale,\
            nfilt,filt,vfilt,ncalc,itype,nintp,nrealp,nchp,icald,rcald = read_drv(runname)
            
        MODIFICATION HISTORY : Juan Alday (29/09/2019)
            
    """

    f = open(runname+'.drv','r')
    
    #Reading header
    _ = f.readline().split() # header
    var1 = f.readline().split()
    var2 = f.readline().split()
    _ = f.readline().split() # linkey
    
    #Reading flags
    ###############
    flags = f.readline().split()
    iconv = int(flags[0])
    flagh2p = int(flags[1])
    ncont = int(flags[2])
    flagc = int(flags[3])
    
    #Reading name of .xsc file
    _ = f.readline().split() # xscname1
    
    #Reading variables
    ###################
    var1 = f.readline().split()
    nlayer = int(var1[0])
    npath = int(var1[1])
    ngas = int(var1[2])
    
    gasID = np.zeros([ngas],dtype='int32')
    isoID = np.zeros([ngas],dtype='int32')
    iproc = np.zeros([ngas],dtype='int32')
    for i in range(ngas):
        var1 = f.readline().split()
        var2 = f.readline().split()
        gasID[i] = int(var1[0])
        isoID[i] = int(var2[0])
        iproc[i] = int(var2[1])

    #Reading parameters of each layer
    ##################################
    _ = f.readline().split() # header
    _ = f.readline().split() # header
    _ = f.readline().split() # header
    _ = f.readline().split() # header
    baseH = np.zeros([nlayer])
    delH = np.zeros([nlayer])
    baseP = np.zeros([nlayer])
    baseT = np.zeros([nlayer])
    totam = np.zeros([nlayer])
    press = np.zeros([nlayer])
    temp = np.zeros([nlayer])
    doppler = np.zeros([nlayer])
    par_coldens = np.zeros([nlayer,ngas])
    par_press = np.zeros([nlayer,ngas])
    cont_coldens = np.zeros([nlayer,ncont])
    hfp = np.zeros([nlayer])
    hfc = np.zeros([nlayer,ncont])
    for i in range(nlayer):
        #Reading layers
        var1 = f.readline().split()
        baseH[i] = float(var1[1])
        delH[i] = float(var1[2])
        baseP[i] = float(var1[3])
        baseT[i] = float(var1[4])
        totam[i] = float(var1[5])
        press[i] = float(var1[6])
        temp[i] = float(var1[7])
        doppler[i] = float(var1[8])

        #Reading partial pressures and densities of gases in each layer
        nlines = ngas*2./6.
        if nlines-int(nlines)>0.0:
            nlines = int(nlines)+1
        else:
            nlines = int(nlines)

        ix = 0
        var = np.zeros([ngas*2])
        for il in range(nlines):
            var1 = f.readline().split()
            for j in range(len(var1)):
                var[ix] = var1[j]
                ix = ix + 1

        ix = 0
        for il in range(ngas):
            par_coldens[i,il] = var[ix]
            par_press[i,il] = var[ix+1]
            ix = ix + 2
        
        #Reading amount of aerosols in each layer
        nlines = ncont/6.
        if nlines-int(nlines)>0.0:
            nlines = int(nlines)+1
        else:
            nlines = int(nlines)
        var = np.zeros([ncont])
        ix = 0
        for il in range(nlines):
            var1 = f.readline().split()
            for j in range(len(var1)):
                var[ix] = var1[j]
                ix = ix + 1

        ix = 0
        for il in range(ncont):
            cont_coldens[i,il] = var[ix]
            ix = ix + 1

        #Reading if FLAGH2P is set
        if flagh2p==1:
            var1 = f.readline().split()
            hfp[i] = float(var1[0])


        #Reading if FLAGC is set
        if flagc==1:
            var = np.zeros([ncont+1])
            ix = 0
            for il in range(ncont):
                var1 = f.readline().split()
                for j in range(len(var1)):
                    var[ix] = var1[j]
                    ix = ix + 1

            ix = 0
            for il in range(ncont):
                hfc[i,il] = var[ix]
                ix = ix + 1

                    
    #Reading the atmospheric paths
    #########################################
    nlayin = np.zeros([npath],dtype='int32')
    imod = np.zeros([npath])
    errlim = np.zeros([npath])
    layinc = np.zeros([npath,2*nlayer],dtype='int32')
    emtemp = np.zeros([npath,2*nlayer])
    scale = np.zeros([npath,2*nlayer])
    for i in range(npath):
        var1 = f.readline().split()
        nlayin[i] = int(var1[0])
        imod[i] = int(var1[1])
        errlim[i] = float(var1[2])
        for j in range(nlayin[i]):
            var1 = f.readline().split()
            layinc[i,j] = int(var1[1]) - 1   #-1 stands for the fact that arrays in python start in 0, and 1 in fortran
            emtemp[i,j] = float(var1[2])
            scale[i,j] = float(var1[3])

    #Reading number of filter profile points
    #########################################
    var1 = f.readline().split()
    nfilt = int(var1[0])
    filt = np.zeros([nfilt])
    vfilt = np.zeros([nfilt])
    for i in range(nfilt):
        var1 = f.readline().split()
        filt[i] = float(var1[0])
        vfilt[i] = float(var1[1])
                            
    _ = f.readline().split() # outfile

    #Reading number of calculations
    ################################
    var1 = f.readline().split()
    ncalc = int(var1[0])
    itype = np.zeros([ncalc],dtype='int32')
    nintp = np.zeros([ncalc],dtype='int32')
    nrealp = np.zeros([ncalc],dtype='int32')
    nchp = np.zeros([ncalc],dtype='int32')
    icald = np.zeros([ncalc,10],dtype='int32')
    rcald = np.zeros([ncalc,10])
    for i in range(ncalc):
        var1 = f.readline().split()
        itype[i] = int(var1[0])
        nintp[i] = int(var1[1])
        nrealp[i] = int(var1[2])
        nchp[i] = int(var1[3])
        for j in range(nintp[i]):
            var1 = f.readline().split()
            icald[i,j] = int(var1[0])
        for j in range(nrealp[i]):
            var1 = f.readline().split()
            rcald[i,j] = float(var1[0])
        for j in range(nchp[i]):
            var1 = f.readline().split()
            #NOT FINISHED HERE!!!!!!

    f.close()

    if MakePlot==True:
        #Plotting the model for the atmospheric layers
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,7))
        ax1.semilogx(baseP,baseH)
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Base altitude (km)')
        ax1.grid()
        
        ax2.plot(baseT,baseH)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Base altitude (km)')
        ax2.grid()
        
        ax3.semilogx(totam,baseH)
        ax3.set_xlabel('Vertical column density in layer (cm$^{-2}$)')
        ax3.set_ylabel('Base altitude (km)')
        ax3.grid()
        
        for i in range(ngas):
            #strgas = spec.read_gasname(gasID[i],isoID[i])
            strgas = 'CHANGE'
            ax4.semilogx(par_coldens[:,i],baseH,label=strgas)
    
        ax4.legend()
        ax4.set_xlabel('Vertical column density in layer (cm$^{-2}$)')
        ax4.set_ylabel('Base altitude (km)')
        ax4.grid()
        
        plt.tight_layout()
        
        plt.show()

    return iconv,flagh2p,ncont,flagc,nlayer,npath,ngas,gasID,isoID,iproc,\
            baseH,delH,baseP,baseT,totam,press,temp,doppler,par_coldens,par_press,cont_coldens,hfp,hfc,\
            nlayin,imod,errlim,layinc,emtemp,scale,\
            nfilt,filt,vfilt,ncalc,itype,nintp,nrealp,nchp,icald,rcald
            
###############################################################################################

def read_inp(runname,Measurement=None,Scatter=None,Spectroscopy=None):

    """
        FUNCTION NAME : read_inp()
        
        DESCRIPTION : Read the .inp file for a Nemesis run

        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            ispace :: (0) Wavenumber in cm-1 (1) Wavelength in um
            iscat :: (0) Thermal emission calculation
                    (1) Multiple scattering required
                    (2) Internal scattered radiation field is calculated first (required for limb-
                        scattering calculations)
                    (3) Single scattering plane-parallel atmosphere calculation
                    (4) Single scattering spherical atmosphere calculation
            ilbl :: (0) Pre-tabulated correlated-k calculation
                    (1) Line by line calculation
                    (2) Pre-tabulated line by line calculation
            
            woff :: Wavenumber/wavelength calibration offset error to be added to the synthetic spectra
            niter :: Number of iterations of the retrieval model required
            philimit :: Percentage convergence limit. If the percentage reduction of the cost function phi
                        is less than philimit then the retrieval is deemed to have converged.
            nspec :: Number of retrievals to perform (for measurements contained in the .spx file)
            ioff :: Index of the first spectrum to fit (in case that nspec > 1).
            lin :: Integer indicating whether the results from previous retrievals are to be used to set any
                    of the atmospheric profiles. (Look Nemesis manual)
        
        CALLING SEQUENCE:
        
            Measurement,Scatter,Spectroscopy,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN = read_inp(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    from archnemesis import Scatter_0, Measurement_0, Spectroscopy_0

    #Getting number of lines 
    nlines = file_lines(runname+'.inp')
    if nlines < 7:
        raise RuntimeError(f"Not enough lines when reading {runname}.inp")
    elif nlines == 7:
        iiform = 0
    elif nlines == 8:
        iiform = 1
    elif nlines > 8:
        iiform = 2

    #Opening file
    f = open(runname+'.inp','r')
    tmp = f.readline().split()
    ispace = WaveUnit(int(tmp[0]))
    iscat = ScatteringCalculationMode(int(tmp[1]))
    ilbl = SpectralCalculationMode(int(tmp[2]))

    if Measurement is None:
        Measurement = Measurement_0()
    Measurement.ISPACE = ispace

    if Scatter is None:
        Scatter = Scatter_0()
    Scatter.ISPACE = ispace
    Scatter.ISCAT = iscat

    if Spectroscopy==None:
        Spectroscopy = Spectroscopy_0(RUNNAME=runname)
    Spectroscopy.ILBL = ilbl

    tmp = f.readline().split()
    WOFF = float(tmp[0])
    fmerrname = str(f.readline().split()[0])
    tmp = f.readline().split()
    NITER = int(tmp[0])
    tmp = f.readline().split()
    PHILIMIT = float(tmp[0])

    tmp = f.readline().split()
    NSPEC = int(tmp[0])
    IOFF = int(tmp[1])

    tmp = f.readline().split()
    LIN = int(tmp[0])

    if iiform == 1:
        tmp = f.readline().split()
        iform = SpectraUnit(int(tmp[0]))
        Measurement.IFORM = iform
    elif iiform == 2:
        tmp = f.readline().split()
        iform = SpectraUnit(int(tmp[0]))
        Measurement.IFORM = iform
        tmp = f.readline().split()
        Measurement.V_DOPPLER = float(tmp[0])
    else:
        Measurement.IFORM = SpectraUnit.Radiance

    return Measurement, Scatter, Spectroscopy, WOFF, fmerrname, NITER, PHILIMIT, NSPEC, IOFF, LIN

###############################################################################################

def read_set(runname,Layer=None,Surface=None,Stellar=None,Scatter=None):
    
    """
        FUNCTION NAME : read_set()
        
        DESCRIPTION : Read the .set file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Scatter :: Python class defining the scattering calculations
            Stellar :: Python class defining the stellar properties
            Surface :: Python class defining the surface properties
            Layer :: Python class defining the layering scheme of the atmosphere
        
        CALLING SEQUENCE:
        
            Scatter,Stellar,Surface,Layer = read_set(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    from archnemesis import Scatter_0, Stellar_0, Surface_0

    #Opening file
    f = open(runname+'.set','r')
    dummy = f.readline().split()
    nmu1 = f.readline().split()
    nmu = int(nmu1[5])
    mu = np.zeros([nmu],dtype='d')
    wtmu = np.zeros([nmu],dtype='d')
    for i in range(nmu):
        tmp = np.fromfile(f,sep=' ',count=2,dtype='d')
        mu[i] = tmp[0]
        wtmu[i] = tmp[1]
    
    dummy = f.readline().split()
    nf = int(dummy[5])
    dummy = f.readline().split()
    nphi = int(dummy[8])
    dummy = f.readline().split()
    isol = int(dummy[5])
    dummy = f.readline().split()
    dist = float(dummy[5])
    dummy = f.readline().split()
    lowbc = int(dummy[6])
    dummy = f.readline().split()
    galb = float(dummy[3])
    dummy = f.readline().split()
    tsurf = float(dummy[3])

    dummy = f.readline().split()

    dummy = f.readline().split()
    layht = float(dummy[8])
    dummy = f.readline().split()
    nlayer = int(dummy[5])
    dummy = f.readline().split()
    laytp = int(dummy[3])
    dummy = f.readline().split()
    layint = int(dummy[3])

    #Creating or updating Scatter class
    if Scatter==None:
        Scatter = Scatter_0()
        Scatter.NMU = nmu
        Scatter.NF = nf
        Scatter.NPHI = nphi
        Scatter.calc_GAUSS_LOBATTO()
    else:
        Scatter.NMU = nmu
        Scatter.calc_GAUSS_LOBATTO()
        Scatter.NF = nf
        Scatter.NPHI = nphi

    #Creating or updating Stellar class
    if Stellar==None:
        Stellar = Stellar_0()
        Stellar.DIST = dist
        if isol==True:
            Stellar.SOLEXIST = True
            Stellar.read_sol(runname)
        elif isol==False:
            Stellar.SOLEXIST = False
        else:
            raise ValueError('error reading .set file :: SOLEXIST must be either True or False')

    #Creating or updating Surface class
    if Surface==None:
        Surface = Surface_0()

    Surface.LOWBC = LowerBoundaryCondition(lowbc)
    Surface.GALB = galb
    Surface.TSURF = tsurf

    #Creating or updating Layer class
    if Layer==None:
        Layer = Layer_0()
    
    Layer.LAYHT = layht*1.0e3
    Layer.LAYTYP = LayerType(laytp)
    Layer.LAYINT = LayerIntegrationScheme(layint)
    Layer.NLAY = nlayer

    return Scatter,Stellar,Surface,Layer

###############################################################################################

def read_fla(runname):
    
    """
        FUNCTION NAME : read_fla()
        
        DESCRIPTION : Read the .fla file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            runname :: Name of the Nemesis run
            inormal :: ortho/para-H2 ratio is in equilibrium (0) or normal 3:1 (1)
            iray :: (0) Rayleigh scattering optical depth not included
                    (1) Rayleigh optical depths for gas giant atmosphere
                    (2) Rayleigh optical depth suitable for CO2-dominated atmosphere
                    (>2) Rayleigh optical depth suitable for a N2-O2 atmosphere
            ih2o :: Additional H2O continuum off (0) or on (1)
            ich4 :: Additional CH4 continuum off (0) or on (1)
            io3 :: Additional O3 continuum off (0) or on (1)
            inh3 :: Additional NH3 continuum off (0) or on (1)
            iptf :: Normal partition function calculation (0) or high-temperature partition
                    function for CH4 for Hot Jupiters
            imie :: Only relevant for scattering calculations. (0) Phase function is computed
                    from the associated Henyey-Greenstein hgphase*.dat files. (1) Phase function
                    computed from the Mie-Theory calculated PHASEN.DAT
            iuv :: Additional flag for including UV cross sections off (0) or on (1)
        
        CALLING SEQUENCE:
        
            inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv = read_fla(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
        """
    
    #Opening file
    f = open(runname+'.fla','r')
    s = f.readline().split()
    inormal = ParaH2Ratio(int(s[0]))
    s = f.readline().split()
    iray = RayleighScatteringMode(int(s[0]))
    s = f.readline().split()
    ih2o = int(s[0])
    s = f.readline().split()
    ich4 = int(s[0])
    s = f.readline().split()
    io3 = int(s[0])
    s = f.readline().split()
    inh3 = int(s[0])
    s = f.readline().split()
    iptf = int(s[0])
    s = f.readline().split()
    imie = AerosolPhaseFunctionCalculationMode(int(s[0]))
    s = f.readline().split()
    iuv = int(s[0])
   
    return inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv

###############################################################################################

def write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv):

    """
        FUNCTION NAME : write_fla()
        
        DESCRIPTION : Write the .fla file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            inormal :: ortho/para-H2 ratio is in equilibrium (0) or normal 3:1 (1)
            iray :: (0) Rayleigh scattering optical depth not included
                    (1) Rayleigh optical depths for gas giant atmosphere
                    (2) Rayleigh optical depth suitable for CO2-dominated atmosphere
                    (>2) Rayleigh optical depth suitable for a N2-O2 atmosphere
            ih2o :: Additional H2O continuum off (0) or on (1)
            ich4 :: Additional CH4 continuum off (0) or on (1)
            io3 :: Additional O3 continuum off (0) or on (1)
            inh3 :: Additional NH3 continuum off (0) or on (1)
            iptf :: Normal partition function calculation (0) or high-temperature partition
                    function for CH4 for Hot Jupiters
            imie :: Only relevant for scattering calculations. (0) Phase function is computed
                    from the associated Henyey-Greenstein hgphase*.dat files. (1) Phase function
                    computed from the Mie-Theory calculated PHASEN.DAT
            iuv :: Additional flag for including UV cross sections off (0) or on (1)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .fla file       
 
        CALLING SEQUENCE:
        
            write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open(runname+'.fla','w')
    f.write('%i \t %s \n' % (int(inormal),'!INORMAL'))
    f.write('%i \t %s \n' % (int(iray),'!IRAY'))
    f.write('%i \t %s \n' % (ih2o,'!IH2O'))
    f.write('%i \t %s \n' % (ich4,'!ICH4'))
    f.write('%i \t %s \n' % (io3,'!IO3'))
    f.write('%i \t %s \n' % (inh3,'!INH3'))
    f.write('%i \t %s \n' % (iptf,'!IPTF'))
    f.write('%i\t %s \n' % (int(imie),'!IMIE'))
    f.write('%i\t %s \n' % (iuv,'!IUV'))
    f.close()

###############################################################################################

def write_set(runname,nmu,nf,nphi,isol,dist,lowbc,galb,tsurf,layht,nlayer,laytp,layint):

    """
        FUNCTION NAME : write_set()
        
        DESCRIPTION : Read the .set file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            nmu :: Number of zenith ordinates
            nf :: Required number of Fourier components
            nphi :: Number of azimuth angles
            isol :: Sunlight on/off
            dist :: Solar distance (AU)
            lowbc :: Lower boundary condition (0 Thermal - 1 Lambertian)
            galb :: Ground albedo
            tsurf :: Surface temperature (if planet is not gasgiant)
            layht :: Base height of lowest layer
            nlayer :: Number of vertical levels to split the atmosphere into
            laytp :: Flag to indicate how layering is perfomed (radtran)
            layint :: Flag to indicate how layer amounts are calculated (radtran)

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .set file

        CALLING SEQUENCE:
        
            l = write_set(runname,nmu,nf,nphi,isol,dist,lowbc,galb,tsurf,layht,nlayer,laytp,layint)
        
        MODIFICATION HISTORY : Juan Alday (15/10/2019)
        
    """

    #Calculating the Gauss-Lobatto quadtrature points
    iScatter = Scatter_0(NMU=nmu)

    #Writin the .set file
    f = open(runname+'.set','w')
    f.write('********************************************************* \n')
    f.write('Number of zenith angles : '+str(nmu)+' \n')
    for i in range(nmu):
        f.write('\t %10.12f \t %10.12f \n' % (iScatter.MU[i],iScatter.WTMU[i]))
    f.write('Number of Fourier components : '+str(nf)+' \n')
    f.write('Number of azimuth angles for fourier analysis : '+str(nphi)+' \n')
    f.write('Sunlight on(1) or off(0) : '+str(isol)+' \n')
    f.write('Distance from Sun (AU) : '+str(dist)+' \n')
    f.write('Lower boundary cond. Thermal(0) Lambert(1) : '+str(lowbc)+' \n')
    f.write('Ground albedo : '+str(galb)+' \n')
    f.write('Surface temperature : '+str(tsurf)+' \n')
    f.write('********************************************************* \n')
    f.write('Alt. at base of bot.layer (not limb) : '+str(layht)+' \n')
    f.write('Number of atm layers : '+str(nlayer)+' \n')
    f.write('Layer type : '+str(laytp)+' \n')
    f.write('Layer integration : '+str(layint)+' \n')
    f.write('********************************************************* \n')

    f.close()

###############################################################################################

def write_inp(runname,ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin,IFORM=-1):

    """
        FUNCTION NAME : write_inp()
        
        DESCRIPTION : Write the .inp file for a Nemesis run
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            ispace :: (0) Wavenumber in cm-1 (1) Wavelength in um
            iscat :: (0) Thermal emission calculation
                    (1) Multiple scattering required
                    (2) Internal scattered radiation field is calculated first (required for limb-
                        scattering calculations)
                    (3) Single scattering plane-parallel atmosphere calculation
                    (4) Single scattering spherical atmosphere calculation
            ilbl :: (0) Pre-tabulated correlated-k calculation
                    (1) Line by line calculation
                    (2) Pre-tabulated line by line calculation
            woff :: Wavenumber/wavelength calibration offset error to be added to the synthetic spectra
            niter :: Number of iterations of the retrieval model required
            philimit :: Percentage convergence limit. If the percentage reduction of the cost function phi
                        is less than philimit then the retrieval is deemed to have converged.
            nspec :: Number of retrievals to perform (for measurements contained in the .spx file)
            ioff :: Index of the first spectrum to fit (in case that nspec > 1).
            lin :: Integer indicating whether the results from previous retrievals are to be used to set any
                    of the atmospheric profiles. (Look Nemesis manual)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
        CALLING SEQUENCE:

            write_inp(runname,ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin,IFORM=iform)
         
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    #Opening file
    f = open(runname+'.inp','w')
    f.write('%i \t %i \t %i \n' % (int(ispace), int(iscat), int(ilbl)))
    f.write('%10.5f \n' % (woff))
    f.write(runname+'.err \n')
    f.write('%i \n' % (niter))
    f.write('%10.5f \n' % (philimit))
    f.write('%i \t %i \n' % (nspec, ioff))
    f.write('%i \n' % (lin))
    if IFORM != -1:
        f.write('%i \n' % (int(IFORM)))
    f.close()


###############################################################################################

def read_pre(runname):
    
    
    """
        FUNCTION NAME : read_pre()

        DESCRIPTION :
        
            Reads the .pre file for a Nemesis run.
            
            This file includes information about the retrieved state vector and the retrieved
            covariance matrix. This file is generated as an output file in the retrieval. 
            If it is copied to a .pre extension and LIN>0 in the .inp file, this file will be used
            to implement information from previous retrievals into the new retrieval
            
            NOTE :: The implementation of this file in archNEMESIS is slightly different
                    than in NEMESIS, but the purpose of the file remains the same.
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Variables_prev :: Variables class with the information from the previous retrieval
            
        
        CALLING SEQUENCE:

            Variables_prev = read_pre(runname)

        MODIFICATION HISTORY : Juan Alday (29/09/2025)
        
    """
    
    #Reading all lines first
    with open(runname + '.pre', 'r') as f: 
        # Skip all lines that do not start with '#' 
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    #Reading file
    idx = 0
    nspec = int(lines[idx].split()[0])
    _lgr.debug(f'nspec = {nspec}')
    idx += 1
    if nspec != 1:
        raise ValueError('error in read_raw :: nspec>1 and this has not yet been implemented in archnemesis')
    
    for ispec in range(nspec):
    
        #Reading lines
        ispecx = int(lines[idx].split()[0])
        idx += 1
        
        #Reading latitude and longitude
        lat = float(lines[idx].split()[0])
        lon = float(lines[idx].split()[1])
        idx += 1
        _lgr.debug(f'lat = {lat}, lon = {lon}')
        
        #Reading other parameters
        npro = int(lines[idx].split()[0])
        ngas = int(lines[idx].split()[1])
        ndust = int(lines[idx].split()[2])
        nlocations = int(lines[idx].split()[3])
        nvar = int(lines[idx].split()[4])
        idx += 1
        _lgr.debug(f'npro = {npro}, ngas = {ngas}, ndust = {ndust}, nlocations = {nlocations}, nvar = {nvar}')
        
        #Reading variable ids
        varident = np.zeros((nvar,3),dtype='int32')
        varparam = np.zeros((nvar,5))
        for ivar in range(nvar):
            
            idx += 1  # Skipping header line
            
            for j in range(3):
                varident[ivar,j] = int(lines[idx].split()[j])
            idx += 1
            
            for j in range(5):
                varparam[ivar,j] = float(lines[idx].split()[j])
            idx += 1
            
            _lgr.debug(f'varident[{ivar}] = {varident[ivar,:]}, varparam[{ivar}] = {varparam[ivar,:]}')
            
        #Reading state vector
        nx = int(lines[idx].split()[0])
        idx += 1
        _lgr.debug(f'nx = {nx}')
        
        xn = np.zeros(nx)
        lx = np.zeros(nx,dtype='int32')
        num = np.zeros(nx,dtype='int32')
        for ix in range(nx):
            xn[ix] = float(lines[idx].split()[0])
            lx[ix] = int(lines[idx].split()[1])
            num[ix] = int(lines[idx].split()[2])
            idx += 1
            
        #Reading the retrieved coviariance matrix
        st = np.zeros((nx,nx))
        for ix in range(nx):
            for iy in range(nx):
                st[ix,iy] = float(lines[idx].split()[0])
                idx += 1
            
    #Creating variables class 
    Variables_prev = ans.Variables_0()    
    Variables_prev._models = []
    jsurf = -1
    jalb = -1
    jxsc = -1
    jtan = -1
    jpre = -1
    #jrad = -1
    jlogg = -1
    jfrac = -1
    ix = 0
    for ivar in range(nvar):
        
        for model in Models:
            
            if model.is_varident_valid(varident[ivar,:]):
                found_model_for_varident = True
                
                try:
                    Variables_prev._models.append(
                        model.from_bookmark(
                            Variables_prev, 
                            varident[ivar,:], 
                            varparam[ivar,:], 
                            ix, 
                            npro, 
                            ngas,
                            ndust,
                            nlocations,
                        )
                    )
                except Exception as e:
                    raise ValueError(f'Failed to read {ivar}^th model entry (with VARIDENT={varident[ivar,:]})') from e
                
                _lgr.info(f'\nVariables_0 :: read_raw :: varident {varident[ivar,:]}. Constructed model "{model.__name__}" (id={model.id})')
                
                if varident[ivar][2]==999:  #Retrieval of surface temperature
                    jsurf = ix
                elif varident[ivar][2]==666: #Retrieval of pressure at a givent tangent height
                    jpre = ix
                elif varident[ivar][2]==777: #Retrieval of tangent height at a given pressure level
                    jtan = ix
                
        ix += Variables_prev._models[-1].n_state_vector_entries

        if not found_model_for_varident:
            raise ValueError(f'Variables_0 :: read_apr :: no model found for varident {varident[ivar,:]}')

        
    Variables_prev.NVAR=nvar
    Variables_prev.NPARAM=5
    Variables_prev.edit_VARIDENT(varident)
    Variables_prev.edit_VARPARAM(varparam)
    Variables_prev.calc_NXVAR(npro,nlocations=nlocations)
    Variables_prev.JPRE, Variables_prev.JTAN, Variables_prev.JSURF, Variables_prev.JALB, Variables_prev.JXSC, Variables_prev.JLOGG, Variables_prev.JFRAC = jpre, jtan, jsurf, jalb, jxsc, jlogg, jfrac
    Variables_prev.NX = nx
    Variables_prev.edit_XA(xn)
    Variables_prev.edit_XN(xn)
    Variables_prev.edit_SA(st)
    Variables_prev.edit_LX(lx)
    Variables_prev.NUM = num
    Variables_prev.calc_DSTEP()
    Variables_prev.calc_FIX()
    
    return Variables_prev


###############################################################################################

def read_fmerr(filename):
    
    
    """
        FUNCTION NAME : read_fmerr()

        DESCRIPTION :
        
            Reads the forward modelling error from a file
        
        INPUTS :
        
            filename :: Name of the file containing the forward modelling error
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            verr(nwave) :: Wavenumber/Wavelength array
            fwerr(nwave) :: Forward modelling error
        
        CALLING SEQUENCE:

            verr,fwerr = read_fmerr(filename)

        MODIFICATION HISTORY : Juan Alday (29/09/2025)
        
    """
    
    with open(filename, 'r') as f:
        
        for line in f:
            if line.strip() and not line.startswith('#'):
                nwave = int(line.split()[0])
                break
            
        verr = np.zeros(nwave)
        fwerr = np.zeros(nwave)
        i = 0
        for line in f:
            if line.strip() and not line.startswith('#'):
                values = line.split()
                if len(values) >= 2:
                    verr[i] = float(values[0])
                    fwerr[i] = float(values[1])
                    i += 1
                if i >= nwave:
                    break

    return verr, fwerr

###############################################################################################

def write_err(runname,nwave,wave,fwerr):

    """
        FUNCTION NAME : write_err()
        
        DESCRIPTION : Write the .err file, including information about forward modelling error
        
        INPUTS :
        
            runname :: Name of Nemesis run
            nwave :: Number of wavelengths at which the albedo is defined
            wave(nwave) :: Wavenumber/Wavelength array
            fwerr(nwave) :: Forward modelling error
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemeis .err file       
 
        CALLING SEQUENCE:
        
            write_err(runname,nwave,wave,fwerr)
         
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open(runname+'.err','w')
    f.write('\t %i \n' % (nwave))
    for i in range(nwave):
        f.write('\t %10.5f \t %10.5f \n' % (wave[i],fwerr[i]))
    f.close()

###############################################################################################

def write_fcloud(npro,naero,height,frac,icloud, MakePlot=False):
    
    """
        FUNCTION NAME : write_fcloud()
        
        DESCRIPTION : Writes the fcloud.ref file, which specifies if the cloud is in the form of
                      a uniform thin haze or is instead arranged in thicker clouds covering a certain
                      fraction of the mean area.
        
        INPUTS :
        
            npro :: Number of altitude profiles in reference atmosphere
            naero :: Number of aerosol populations in the atmosphere
            height(npro) :: Altitude (km)
            frac(npro) :: Fractional cloud cover at each level
            icloud(npro,naero) :: Flag indicating which aerosol types contribute to the broken cloud
                                  which has a fractional cloud cover of frac
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            fcloud.ref file
        
        CALLING SEQUENCE:
        
            write_fcloud(npro,naero,height,frac,icloud)
        
        MODIFICATION HISTORY : Juan Alday (16/03/2021)
        
    """

    f = open('fcloud.ref','w')

    f.write('%i \t %i \n' % (npro,naero))
    
    for i in range(npro):
        str1 = str('{0:7.6f}'.format(height[i]))+'\t'+str('{0:7.3f}'.format(frac[i]))
        for j in range(naero):
            str1 = str1+'\t'+str('{0:d}'.format(icloud[i,j]))
            f.write(str1+'\n')

    f.close()




