from __future__ import annotations #  for 3.9 compatability


import os
import os.path
import textwrap
import sys
from typing import Type, Iterable

import numpy as np
import matplotlib.pyplot as plt

from archnemesis import *
from archnemesis.Models import Models, ModelBase, ModelParameterEntry
from archnemesis.enums import AtmosphericProfileType, Gas
from archnemesis.helpers import io_helper
from archnemesis.helpers import h5py_helper

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

Model variables Class.
"""

class AprReadError(RuntimeError):
    """
    Something has gone wrong when reading a *.apr file
    """


class Variables_0:

    def __init__(self, NVAR=2, NPARAM=10, NX=10, JPRE=-1, JTAN=-1, JSURF=-1, JALB=-1, JXSC=-1, JRAD=-1, JLOGG=-1, JFRAC=-1):

        """
        Inputs
        ------
        @param NVAR: int,
            Number of model variables to be included
        @param NPARAM: int,
            Number of extra parameters needed to implement the different models       
        @param NX: int,
            Number of points in the state vector
        @param JPRE: int,
            Position of ref. tangent pressure in state vector (if included)
        @param JTAN: int,
            Position of tangent altitude correction in state vector (if included)
        @param JSURF: int,
            Position of surface temperature in state vector (if included)
        @param JALB: int,
            Position of start of surface albedo spectrum in state vector (if included)
        @param JXSC: int,
            Position of start of x-section spectrum in state vector (if included)
        @param JRAD: int,
            Position of radius of the planet in state vector (if included)
        @param JLOGG: int,
            Position of surface log_10(g) of planet in state vector (if included)     
         @param JFRAC: int,
            Position of fractional coverage in state vector (if included)  

        Attributes
        ----------
        @attribute VARIDENT: 2D array (NVAR,3)
            Variable ID
        @attribute VARPARAM: 2D array (NVAR,NPARAM)
            Extra parameters needed to implement the parameterisation
        @attribute NXVAR: 1D array
            Number of points in state vector associated with each variable
        @attribute XA: 1D array
            A priori State vector
        @attribute SA: 2D array
            A priori Covariance matrix of the state vector
        @attribute XN: 1D array
            State vector
        @attribute SX: 2D array
            Covariance matrix of the state vector
        @attribute LX: 1D array
            Flag indicating whether the elements of the state vector are carried in log-scale
        @attribute FIX: 1D array
            Flag indicating whether the elements of the state vector must be fixed
        @attribute NUM: 1D array
            Flag indicating how the gradients with respect to a particular element of the state vector must be computed
            (0) Gradients are computed analytically inside CIRSradg (Atmospheric gradients or Surface temperature) or subspecretg (Others)
            (1) Gradients are computed numerically 
        @attribute DSTEP: 1D array
            For the elements of the state vector whose derivative is being calculated numerically, this array indicates
            the step in the function to be used to calculate the numerical derivative (f' = (f(x+h) - f(x)) / h ). 
            If not specified, this step is assumed to be 5% of the value (h = 0.05*x)

        Methods
        -------
        Variables_0.edit_VARIDENT()
        Variables_0.edit_VARPARAM()
        Variables_0.edit_XA()
        Variables_0.edit_XN()
        Variables_0.edit_LX()
        Variables_0.edit_SA()
        Variables_0.edit_SX()
        Variables_0.calc_NXVAR()
        Variables_0.calc_DSTEP()
        Variables_0.calc_FIX()
        Variables_0.read_hdf5()
        Variables_0.read_apr()
        """

        #Input parameters
        self.NVAR = NVAR
        self.NPARAM = NPARAM
        self.NX = NX
        self.JPRE = JPRE
        self.JTAN = JTAN
        self.JSURF = JSURF
        self.JALB = JALB
        self.JXSC = JXSC
        self.JRAD = JRAD
        self.JLOGG = JLOGG
        self.JFRAC = JFRAC
        
        
        # Input the following profiles using the edit_ methods.
        self.VARIDENT = None # np.zeros(NVAR,3)
        self.VARPARAM = None # np.zeros(NVAR,NPARAM)
        self.NXVAR =  None # np.zeros(NX)
        self.XN = None # np.zeros(NX)
        self.LX = None # np.zeros(NX)
        self.FIX =  None # np.zeros(NX)
        self.SX = None # np.zeros((NX, NX))
        self.NUM = None #np.zeros(NX)
        self.DSTEP = None #np.zeros(NX)
        
        # private attributes
        self._models = None
    ################################################################################################################

    @property
    def models(self) -> list[ModelBase]:
        """
        Returns a list filled with the models whose parameters are in the state vector
        """
        
        if self._models is None:
            raise AttributeError(f'Models have not been found yet (e.g. by "Variables_0.read_apr(...)")')
        return self._models
    
    @property
    def gas_vmr_models(self) -> Iterable[ModelBase]:
        """
        Returns an iterable of models that deal with gas volume mixing ratios
        """
        return filter(lambda model: hasattr(model, 'target') and model.target == AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO, self.models)
    
    @property
    def aerosol_density_models(self) -> Iterable[ModelBase]:
        """
        Returns an iterable of models that deal with aerosol densities
        """
        return filter(lambda model: hasattr(model, 'target') and model.target == AtmosphericProfileType.AEROSOL_DENSITY, self.models)
    
    
    @property
    def model_parameters(self) -> tuple[dict[str,ModelParameterEntry],...]:
        """
        Returns a tuple of dictionaries that contain the values of the parameters of the models in the state vector
        
        
        ## RETURNS ##
        
        all_model_parameters : tuple[dict[str, ModelParameterEntry], ...]
            A tuple of dictionaries that map model parameter names to the "ModelParameterEntry" for that parameter.
        """
        return tuple(
            model.get_parameters_from_state_vector(
                self.XA,
                self.XN,
                self.LX,
                self.FIX
            )
            for model in self.models
        )
    
    
    def model_parameters_as_string(
            self,
        ) -> str:
        """
        Returns a string of the model parameters (formatted as a table)
        """
        mp = self.model_parameters
        
        
        # General header for output, not part of the table, but used for explanatory information
        p_hdr = '\n'.join((
            'i - index of model parameter in the state vector',
        )) + '\n'
        
        # Header for the table, i.e. column labels
        p_tbl_hdr = ('i', 'model id', 'parameter name', 'apriori value', 'posterior value', 'apriori error')
        p_tbl_col_widths = [len(x) for x in p_tbl_hdr]
        p_tbl_full_col_width = None
        p_str = []
        for i in range(len(mp)+1):
            if i==0:
                p_str.append([p_tbl_hdr])
                continue
            else:
                p_str.append([])
            first1 = True
            for p_name, p_val in mp[i-1].items():
                is_log = self.LX[p_val.sv_slice] == 1
                apriori_error = np.sqrt(np.diag(self.SA[p_val.sv_slice,p_val.sv_slice]))
                apriori_error = np.where(is_log, apriori_error*p_val.apriori_value, apriori_error)
                more_than_one_entry = len(p_val.apriori_value) > 1
                for j, (fix, a, b, s) in enumerate(zip(p_val.is_fixed, p_val.apriori_value, p_val.posterior_value, apriori_error)):
                    
                    p_str[i].append((
                        f'{p_val.sv_slice.start+j}',
                        f'{p_val.model_id}' if first1 else '---', 
                        p_name+f'[{j}]' if more_than_one_entry else p_name, 
                        f'{a:07.2E}', 
                        'FIXED' if fix else f'{b:07.2E}',
                        'FIXED' if fix else f'{s:07.2E}'
                    ))
                    first1 = False
                    first2 = False
                    p_tbl_col_widths = [max(len(_1), _2) for _1,_2 in zip(p_str[i][-1], p_tbl_col_widths)]
        
        tbl_sec_sep = '|-' + ('-|-'.join(('-'*w for w in p_tbl_col_widths))) + '-|'
        p_tbl_full_col_width = len(tbl_sec_sep)
        
        for i in range(len(p_str)):
            for j in range(len(p_str[i])):
                p_str[i][j] = '| ' + (' | '.join((x.ljust(p_tbl_col_widths[k], ' ') for k,x in enumerate(p_str[i][j])))) + ' |'
            p_str[i] = '\n'.join(p_str[i])
        
        p_str = (f'\n{tbl_sec_sep}\n').join(p_str)
        
        return '\n'.join((
            p_hdr,
            '-'*p_tbl_full_col_width,
            p_str,
            '-'*p_tbl_full_col_width,
        ))
    
    
    def plot_model_parameters(
            self, 
            plot_dir : None | str = None, 
            show : bool = False
        ):
        """
        Plots the parameters of the models in the state vector
        
        ARGUMENTS
            plot_dir : None | str = None
                if not 'None', will save the plot to this directory to save the plot to
                this directory with the file name 'model_parameters'
            show : bool = False
                if True will show the plot interactively
        """
        print(self.model_parameters_as_string())
        
        mp = self.model_parameters
        
        _lgr.debug('all model parameters:')
        for x in mp:
            _lgr.debug(f'\t{x}')
        
        n_rows = len(mp) # number of models we have
        n_cols = 10 # this is the 'resolution' of columns
        
        # Work out how much space each parameter of each model needs on its row
        # is in fraction of row used up by a parameter. I.e. if i^th model has two parameters (p1,p2)
        # and p1 has 2 points, and p2 has 3 points, then frac[i] = (0.4, 0.6)
        f_cols = tuple(
            tuple(
                (v.sv_slice.stop - v.sv_slice.start)
                /(
                    sum(
                        (u.sv_slice.stop-u.sv_slice.start for u in x.values())
                    )
                ) for v in x.values()
            ) for x in mp
        )
        _lgr.debug(f'{n_rows=} {n_cols=}')
        _lgr.debug(f'{f_cols=}')
        
        
        subplot_layout = [['' for c in range(n_cols)] for r in range(n_rows)]
        _lgr.debug(f'subplot_layout.shape = ({len(subplot_layout)}, {len(subplot_layout[0])})')
        
        for r, model_params in enumerate(mp):
            c0 = 0
            c = 0
            f_col = f_cols[r]
            for p_idx, (p_name, p_val) in enumerate(model_params.items()):
                last_col = c0 + f_col[p_idx] * n_cols
                model_param_id_str = f'{p_val.model_id}_{r}_{p_idx}'
                _lgr.debug(f'{model_param_id_str}')
                
                while c < last_col:
                    _lgr.debug(f'{r=} {c=}')
                    _lgr.debug(f'before: {subplot_layout[r][c]=}')
                    subplot_layout[r][c] = model_param_id_str
                    _lgr.debug(f'after: {subplot_layout[r][c]=}')
                    c += 1
                
                c0 = last_col
        
        
        _lgr.debug('subplot_layout:')
        for x in subplot_layout:
            _lgr.debug(f'\t{x}')
            
        
        
        
        f = plt.figure(figsize=(8,12))
        
        axes = f.subplot_mosaic(
            subplot_layout, 
            gridspec_kw=dict(
                wspace=0.2*n_cols, 
                hspace=0.1*n_rows
            )
        )
        
        _lgr.debug(f'{len(axes.values())=}')
        _lgr.debug(f'{tuple(axes.keys())=}')
        
        ax_iter = iter(axes.values())
        
        f.suptitle('Model Parameters\n(in same order as varident)', fontsize=10)
        
        is_first_plot=True
        for model_params in mp:
            is_first_col = True
            for pname, pval in model_params.items():
                ax = next(ax_iter)
                x = range(pval.apriori_value.size)
                
                if is_first_col:
                    is_first_col = False
                    ax.set_title(f'{pval.model_id} : {pname}', fontsize=8)
                else:
                    ax.set_title(pname, fontsize=8)
                
                # draw lines
                ax.plot(x, pval.apriori_value, linestyle='-', linewidth=1, marker='none', color='tab:blue', alpha=0.6)
                ax.plot(x, pval.posterior_value, linestyle='-', linewidth=1, marker='none', color='tab:orange', alpha=0.6)
                
                # draw fixed
                temp = np.array(pval.apriori_value)
                temp[~pval.is_fixed] = np.nan
                ax.plot(x, temp, linestyle='none', linewidth=1, marker='x', color='black', alpha=0.6, label='fixed' if is_first_plot else None)
                
                # draw non-fixed
                temp = np.array(pval.apriori_value)
                temp[pval.is_fixed] = np.nan
                ax.plot(x, temp, linestyle='none', linewidth=1, marker='o', color='tab:blue', alpha=0.6, label='apriori' if is_first_plot else None)
                
                temp = np.array(pval.posterior_value)
                temp[pval.is_fixed] = np.nan
                ax.plot(x, temp, linestyle='none', linewidth=1, marker='o', color='tab:orange', alpha=0.6, label='posterior' if is_first_plot else None)
                
                
                # set plot element styling
                ax.tick_params(axis='both', which='both', labelsize=6)
                
                is_first_plot = False
            
            f.legend()
        
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'model_parameters'))
        
        if show:
            plt.show()
        
        return
    
    
    def edit_VARIDENT(self, VARIDENT_array):
        """
        Edit the Variable IDs
        @param VARIDENT_array: 2D array
            Parameter IDs defining the parameterisation
        """
        VARIDENT_array = np.array(VARIDENT_array)
        #assert len(VARIDENT_array[:,0]) == self.NVAR, 'VARIDENT should have (NVAR,3) elements'
        #assert len(VARIDENT_array[0,:]) == 3, 'VARIDENT should have (NVAR,3) elements'
        self.VARIDENT = VARIDENT_array

    ################################################################################################################

    def edit_VARPARAM(self, VARPARAM_array):
        """
        Edit the extra parameters needed to implement the parameterisations
        @param VARPARAM_array: 2D array
            Extra parameters defining the model
        """
        VARPARAM_array = np.array(VARPARAM_array)
        #assert len(VARPARAM_array[:,0]) == self.NVAR, 'VARPARAM should have (NVAR,NPARAM) elements'
        #assert len(VARPARAM_array[0,:]) == self.NPARAM, 'VARPARAM should have (NVAR,NPARAM) elements'
        self.VARPARAM = VARPARAM_array

    ################################################################################################################

    def edit_XA(self, XA_array):
        """
        Edit the State Vector.
        @param XA_array: 1D array
            Parameters defining the a priori state vector
        """
        XA_array = np.array(XA_array)
        assert len(XA_array) == self.NX, 'XA should have NX elements'
        self.XA = XA_array

    ################################################################################################################

    def edit_XN(self, XN_array):
        """
        Edit the State Vector.
        @param XN_array: 1D array
            Parameters defining the state vector
        """
        XN_array = np.array(XN_array)
        assert len(XN_array) == self.NX, 'XN should have NX elements'
        self.XN = XN_array

    ################################################################################################################

    def edit_LX(self, LX_array):
        """
        Edit the the flag indicating if the elements are in log-scale
        @param LX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is in log-scale (1) or not (0)
        """
        LX_array = np.array(LX_array,dtype='int32')
        assert len(LX_array) == self.NX, 'LX should have NX elements'
        self.LX = LX_array  

    ################################################################################################################

    def edit_FIX(self, FIX_array):
        """
        Edit the the flag indicating if the elements are to be fixed
        @param FIX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is fixed (1) or not (0)
        """
        FIX_array = np.array(FIX_array,dtype='int32')
        assert len(FIX_array) == self.NX, 'FIX should have NX elements'
        self.FIX = FIX_array 

    ################################################################################################################

    def edit_SA(self, SA_array):
        """
        Edit the a priori covariance matrix
        @param SA_array: 2D array
            A priori covariance matrix
        """
        SA_array = np.array(SA_array)
        assert len(SA_array[:,0]) == self.NX, 'SA should have (NX,NX) elements'
        assert len(SA_array[0,:]) == self.NX, 'SA should have (NX,NX) elements'
        self.SA = SA_array 

    ################################################################################################################

    def edit_SX(self, SX_array):
        """
        Edit the state vector covariance matrix
        @param SX_array: 2D array
            State vector covariance matrix
        """
        SX_array = np.array(SX_array)
        assert len(SX_array[:,0]) == self.NX, 'SX should have (NX,NX) elements'
        assert len(SX_array[0,:]) == self.NX, 'SX should have (NX,NX) elements'
        self.SX = SX_array 

    ################################################################################################################

    def calc_NXVAR(self, NPRO, nlocations=1):
        """
        Calculate the array defining the number of parameters in the state 
        vector associated with each model
        @param NXVAR_array: 1D array
            Number of parameters in the state vector associated with each model
        """
        
        assert len(self.models) == self.NVAR, "Must have one model per variable"
        
        
        nxvar = np.zeros(len(self.models),dtype='int32')
        assert self.VARIDENT.ndim == 2, "self.VARIDENT must have 2 dimensions"
        
        for i, model in enumerate(self.models):
            nxvar[i] = model.n_state_vector_entries
        
        
        self.NXVAR = nxvar

    ################################################################################################################

    def calc_DSTEP(self):
        """
        Calculate the step size to be used for the calculation of the numerical derivatives
        f' =  ( f(x+h) - f(x) ) / h
        """

        #Generally, we use h = 0.05*x
        dstep = np.zeros(self.NX)
        dstep[:] = self.XN * 0.05

        #Changing the step size for certain parameterisations
        ix = 0
        for i in range(self.NVAR):

            if self.NVAR>1:
                imod = self.VARIDENT[i,2]
                ipar = self.VARPARAM[i,0]
            else:
                imod = self.VARIDENT[0,2]
                ipar = self.VARPARAM[0,0]

            if imod == 228:
                
                #V0,C0,C1,C2,P0,P1,P2,P3
                dstep[ix] = 0.5 * np.sqrt( self.SA[ix,ix] )
                dstep[ix+1] = 0.5 * np.sqrt( self.SA[ix+1,ix+1] )
                dstep[ix+2] = 0.5 * np.sqrt( self.SA[ix+2,ix+2] )
                dstep[ix+3] = 0.5 * np.sqrt( self.SA[ix+3,ix+3] )

            ix = ix + self.NXVAR[i]

        self.DSTEP = dstep

    ################################################################################################################

    def calc_FIX(self):
        """
        Check if the fractional error on any of the state vector parameters is so small 
        that it must be kept constant in the retrieval
        @param FIX: 1D array
            Flag indicating the elements of the state vector that need to be fixed
        """

        minferr = 1.0e-6  #minimum fractional error to fix variable.

        ifix = np.zeros(self.NX,dtype='int32')    
        for ix in range(self.NX):
            xa1 = self.XA[ix]
            ea1 = np.sqrt(abs(self.SA[ix,ix]))

            if self.LX[ix]==1:
                xa1 = np.exp(xa1)
                ea1 = xa1*ea1

            ferr = abs(ea1/xa1)
            if ferr<=minferr:
                ifix[ix] = 1
                
        self.FIX = ifix

    ################################################################################################################

    @staticmethod
    def classify_model_type_from_varident(
            varident : np.ndarray[[3],int],
            ngas : int,
            ndust : int
        ) -> tuple[Type, None | AtmosphericProfileType]:
        """
        Works out the type of model (and subtype if applicable) identified by a VARIDENT triplet.
        
        ## ARGUMENTS ##
            
            varident : np.ndarray[[3],int]
                Three integers that identify a model
                
            ngas : int
                The number of gases present in the reference atmosphere
            
            ndust : int
                The number of aerosol species present in the reference atmosphere
            
        ## RETURNS ##
        
            model_classification : tuple[Type, None | AtmosphericProfileType]
                A tuple containing values that classify the model. From broadest scope to narrowest.
                Currently the tuple has the elements (in order):
                    
                    ModelClass : Type
                        A subclass of archnemesis.Models.ModelBase.ModelBase that denotes the broadest
                        classification of the model. This broadly corresponds to the retrieval component
                        that the model interacts with (e.g. Atmosphere_0, Scatter_0, Measurement_0).
                    
                    ParameterisedTarget : None | AtmosphericProfileType
                        The part of the retrieval component that the model parameterises (and therefore
                        alters). This is 'None' when unknown, or an ENUM corresponding to an attribute
                        of the retrieval component that the model parameterises.
        """
        model_classification = None
        if varident[0] == 0:
            model_classification = ( ModelBase, AtmosphericProfileType.TEMPERATURE)
        elif (varident[0] > 0) and int(varident[0]) in iter(Gas):
            model_classification = ( ModelBase, AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO)
        elif (varident[0] < 0) and (-varident[0]) <= ndust:
            model_classification = ( ModelBase, AtmosphericProfileType.AEROSOL_DENSITY)
        elif (varident[0] < 0) and (-varident[0]) == ndust + 1:
            model_classification = ( ModelBase, AtmosphericProfileType.PARA_H2_FRACTION)
        elif (varident[0] < 0) and (-varident[0]) == ndust + 2:
            model_classification = ( ModelBase, AtmosphericProfileType.FRACTIONAL_CLOUD_COVERAGE)
        else:
            # Other models are classified by their ID number
            model_id_parent_classes = Models[varident[2]].__bases__
            assert len(model_id_parent_classes) == 1, "Only support single inheritance of model classes for now"
            model_classification = (model_id_parent_classes[0],None)
        
        return model_classification
    
    ################################################################################################################

    def read_hdf5(self,runname,npro):
        """
        Read the Variables field of the HDF5 file, which contains information about the variables and
        parametrisations that are to be retrieved, as well as their a priori values.
        These parameters are then included in the Variables class.
        
        N.B. In this code, the apriori and retrieved vectors x are usually
        converted to logs, all except for temperature and fractional scale heights
        This is done to reduce instabilities when different parts of the
        vectors and matrices hold vastly different sized properties. e.g.
        cloud x-section and base height.

        @param runname: str
            Name of the Nemesis run
        @param NPRO: int
            Number of altitude levels in the reference atmosphere
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Variables exists
        e = "/Variables" in f
        if e==False:
            raise ValueError('error :: Variables is not defined in HDF5 file')
        else:

            self.NVAR = h5py_helper.retrieve_data(f, 'Scatter/NVAR', np.int32)
            
    ################################################################################################################

    def read_apr(self,runname,npro,ngas,ndust,nlocations=1):
        """
        Read the .apr file, which contains information about the variables and
        parametrisations that are to be retrieved, as well as their a priori values.
        These parameters are then included in the Variables class.
        
        N.B. In this code, the apriori and retrieved vectors x are usually
        converted to logs, all except for temperature and fractional scale heights
        This is done to reduce instabilities when different parts of the
        vectors and matrices hold vastly different sized properties. e.g.
        cloud x-section and base height.

        Inputs
        ---------------

        @param runname: str
            Name of the Nemesis run
        @param npro: int
            Number of altitude levels in the reference atmosphere
        @param ngas: int
            Number of gasses in the reference atmosphere
        @param ndust: int
            Number of aerosol species in the reference atmosphere
            
        Optional inputs
        ----------------
        
        @param NLOCATIONS: int
            Number of locations in the reference atmosphere/surface
        
        """
        
        from archnemesis import Scatter_0
        
        apr_read_warning_msg = ''
        
        if self._models is not None:
            _lgr.warning(f'Already have models for {runname}, will overwrite them as we read the *.apr file.')
        self._models = []
            

        #Open file
        with open(runname+'.apr','r') as f:
    
            #Reading header
            s = f.readline().split()
        
            #Reading first line
            s = f.readline().split()
            nvar = int(s[0])
        
            #Initialise some variables
            jsurf = -1
            jalb = -1
            jxsc = -1
            jtan = -1
            jpre = -1
            jrad = -1
            jlogg = -1
            jfrac = -1
            sxminfac = 0.001
            mparam = 500        #Giving big sizes but they will be re-sized
            mx = 10000
            varident = np.zeros([nvar,3],dtype='int')
            varparam = np.zeros([nvar,mparam])
            lx = np.zeros([mx],dtype='int')
            x0 = np.zeros([mx])
            sx = np.zeros([mx,mx])
            inum = np.zeros([mx],dtype='int')
            
            #Reading data
            i = 0
            ix = 0
            keep_reading = True
            
            while keep_reading:
            
            
                a = f.readline()
                
                
                if a.strip() == '':
                    # If we have read an empty line
                    if i >= nvar:
                        # if we are past the expected number of VARIDENT entries, we are happy and should stop
                        keep_reading = False
                        break
                    else:
                        # if we are not past the expected number of VARIDENT entries, we are sad
                        keep_reading = False
                        raise AprReadError(f'Expected {nvar} entries in {f.name} but only read {i} before encountering an empty line.')
                else:
                    # We have read a non-empty line
                    if i >= nvar:
                        # If we are reading past the expected number of VARIDENT entries, we are concerned
                        _lgr.warning(f'Expected {nvar} entries in {f.name}, but have found an {i+1}^th entry. Increasing number of variables, extending arrays, and continuing to read file...')
                        # Enlarge arrays where required
                        nvar += 1
                        varident = np.concatenate((varident, np.zeros([1,3],dtype='int')), axis=0)
                        varparam = np.concatenate((varparam, np.zeros([1,mparam])), axis=0)
                
                
                
                s = a.strip().split()
                if len(s) < 3:
                    raise AprReadError(f'VARIDENT entry must have at least 3 components, but {i}^th entry "{a}" has {len(s)}')
                
                try:
                    varident[i,:] = tuple(map(int, s[:3]))
                except Exception as e:
                    raise AprReadError(f'VARIDENT entry MUST be three (3) integers on a single line, other information beyond that is ignored (as a comment)') from e


                found_model_for_varident = False
                
                for model in Models:
                    
                    if model.is_varident_valid(varident[i]):
                        found_model_for_varident = True
                        
                        try:
                            self._models.append(
                                model.from_apr_to_state_vector(
                                    self, 
                                    f, 
                                    varident[i], 
                                    varparam[i], 
                                    ix, 
                                    lx, 
                                    x0, 
                                    sx,
                                    inum, 
                                    npro, 
                                    ngas,
                                    ndust,
                                    nlocations,
                                    runname,
                                    sxminfac
                                )
                            )
                        except Exception as e:
                            raise AprReadError(f'Failed to read {i}^th model entry (with VARIDENT={varident[i]})') from e
                        
                        _lgr.info(f'\nVariables_0 :: read_apr :: varident {varident[i]}. Constructed model "{model.__name__}" (id={model.id})')
                        try:
                            io_helper.OutWidth.push(io_helper.OutWidth.get() - 2)
                            _lgr.info(textwrap.indent(str(self._models[-1].info(lx,x0)), '  '))
                        finally:
                            io_helper.OutWidth.pop()
                        
                        if varident[i][2]==999:  #Retrieval of surface temperature
                            jsurf = ix
                        elif varident[i][2]==666: #Retrieval of pressure at a givent tangent height
                            jpre = ix
                        elif varident[i][2]==777: #Retrieval of tangent height at a given pressure level
                            jtan = ix
                        
                        ix += self._models[-1].n_state_vector_entries

                
                if not found_model_for_varident:
                    raise AprReadError(f'Variables_0 :: read_apr :: no model found for varident {varident[i]}')
                
                i += 1

        nx = ix
        lx1 = np.zeros(nx,dtype='int32')
        inum1 = np.zeros(nx,dtype='int32')
        xa = np.zeros(nx)
        sa = np.zeros([nx,nx])
        lx1[0:nx] = lx[0:nx]
        inum1[0:nx] = inum[0:nx]
        xa[0:nx] = x0[0:nx]
        sa[0:nx,0:nx] = sx[0:nx,0:nx]

        self.NVAR=nvar
        self.NPARAM=mparam
        self.edit_VARIDENT(varident)
        self.edit_VARPARAM(varparam)
        self.calc_NXVAR(npro,nlocations=nlocations)
        self.JPRE, self.JTAN, self.JSURF, self.JALB, self.JXSC, self.JLOGG, self.JFRAC = jpre, jtan, jsurf, jalb, jxsc, jlogg, jfrac
        self.NX = nx
        self.edit_XA(xa)
        self.edit_XN(xa)
        self.edit_SA(sa)
        self.edit_LX(lx1)
        self.NUM = inum1
        self.calc_DSTEP()
        self.calc_FIX()
        
        ################################################################################################################
