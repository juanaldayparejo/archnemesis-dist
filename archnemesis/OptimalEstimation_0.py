from __future__ import annotations #  for 3.9 compatability

import sys

from archnemesis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from copy import *
import pickle


from archnemesis.enums import WaveUnit, SpectraUnit
from archnemesis.helpers.maths_helper import is_diagonal


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

Optimal Estimation Class. It includes all parameters that are relevant for the retrieval of parameters using
                          the Optimal Estimation formalism
"""

class OptimalEstimation_0:

    @classmethod
    def from_itr(cls, runname):
        """
        Read "runname.itr" file, populate OptimalEstimation object as if it was the output of "coreretOE(...)"
        """
        _lgr.info(f'Creating {cls.__name__} instance from {runname}.itr file')
        # Get the number of lines in the file
        n_lines = 0
        with open(f'{runname}.itr', 'rb') as f:
            n_lines = sum(1 for _ in f)
        
        _lgr.info(f'{n_lines=}')
        
        with open(f'{runname}.itr', 'r') as f:
            nx, ny, niter = map(int, f.readline().strip().split())
            
            _lgr.info(f'{nx=} {ny=} {niter=}')
            
            instance = cls(
                IRET=0, 
                NITER=niter, 
                NX=nx, 
                NY=ny, 
                PHILIMIT=None, # We don't have this information so ignore it
                NCORES=None # we don't have this information so ignore it
            )
            
            # Each iteration has 1 + 2*NX + 4*NY + NX*NY lines
            # Therefore work out how many iterations were written to
            # the file and select the last one
            lines_per_record = (1+2*nx+4*ny+nx*ny)
            n_records_in_file = (n_lines - 1)//lines_per_record
            n_skip_lines = (n_records_in_file -1)*lines_per_record
            
            _lgr.info(f'{lines_per_record=} {n_records_in_file=} {n_skip_lines=}')
            _lgr.info('We want the final state of the iterations, so we want to read the final record in the file.')
            
            for _ in range(n_skip_lines):
                f.readline()
            
            # Now read in the final iteration
            chisq, phi = map(float, f.readline().strip().split())
            _lgr.info(f'{chisq=} {phi=}')
            
            xn_array = np.zeros((nx,))
            xa_array = np.zeros((nx,))
            y_array = np.zeros((ny,))
            se_array = np.zeros((ny,ny)) # only stored the diagonal elements in *.itr file
            yn_prev_array = np.zeros((ny,))
            yn_array = np.zeros((ny,))
            kk_array = np.zeros((ny,nx))
            
            for i in range(nx): xn_array[i] = float(f.readline().strip())
            for i in range(nx): xa_array[i] = float(f.readline().strip())
            for i in range(ny): y_array[i] = float(f.readline().strip())
            for i in range(ny): se_array[i,i] = float(f.readline().strip())
            for i in range(ny): yn_prev_array[i] = float(f.readline().strip())
            for i in range(ny): yn_array[i] = float(f.readline().strip())
            for i in range(nx):
                for j in range(ny): kk_array[j,i] = float(f.readline().strip())
            
            instance.edit_XN(xn_array)
            instance.edit_XA(xa_array)
            instance.edit_Y(y_array)
            instance.edit_SE(se_array)
            instance.edit_YN(yn_array)
            instance.edit_KK(kk_array)
        
        return chisq, phi, yn_prev_array, instance


    def __init__(self, IRET=0, NITER=1, NX=1, NY=1, PHILIMIT=0.1, NCORES=1):

        """
        Inputs
        ------
        @param NITER: int,
            Number of iterations in retrieval 
        @param PHILIMIT: real,
            Percentage convergence limit. If the percentage reduction of the cost function PHI
            is less than philimit then the retrieval is deemed to have converged.
        @param NY: int,
            Number of elements in measurement vector    
        @param NX: int,
            Number of elements in state vector
        @param NCORES: int,
            Number of cores available for parallel computations

        Attributes
        ----------
        @attribute PHI: real
            Current value of the Cost function
        @attribute CHISQ: real
            Current value of the reduced chi-squared
        @attribute Y: 1D array
            Measurement vector
        @attribute SE: 1D array
            Measurement covariance matrix
        @attribute YN: 1D array
            Modelled measurement vector
        @attribute XA: 1D array
            A priori state vector
        @attribute SA: 1D array
            A priori covariance matrix
        @attribute XN: 1D array
            Current state vector
        @attribute KK: 2D array
            Jacobian matrix
        @attribute DD: 2D array
            Gain matrix
        @attribute AA: 2D array
            Averaging kernels        
        @attribute SM: 2D array
            Measurement error covariance matrix
        @attribute SN: 2D array
            Smoothing error covariance matrix
        @attribute ST: 2D array
            Retrieved error covariance matrix (SN+SM)

        Methods
        -------
        OptimalEstimation.read_hdf5()
        OptimalEstimation.write_input_hdf5()
        OptimalEstimation.write_output_hdf5()
        OptimalEstimation.edit_Y()
        OptimalEstimation.edit_SE()
        OptimalEstimation.edit_YN()
        OptimalEstimation.edit_XA()
        OptimalEstimation.edit_SA()
        OptimalEstimation.edit_XN()
        OptimalEstimation.edit_KK()
        OptimalEstimation.calc_gain_matrix()
        OptimalEstimation.plot_bestfit()
        """

        #Input parameters
        self.IRET = IRET
        self.NITER = NITER
        self.NX = NX
        self.NY = NY
        self.PHILIMIT = PHILIMIT      
        self.NCORES = NCORES  

        # Input the following profiles using the edit_ methods.
        self.KK = None #(NY,NX)
        self.DD = None #(NX,NY)
        self.AA = None #(NX,NX)
        self.SM = None #(NX,NX)
        self.SN = None #(NX,NX)
        self.ST = None #(NX,NX)
        self.Y= None #(NY)
        self.YN = None #(NY)
        self.SE = None #(NY,NY)
        self.XA = None #(NX)
        self.SA = None #(NX,NX)
        self.XN = None #(NX)
        
        # Output values, will be calculated when Optimal Estimation is performed
        self.PHI = None
        self.CHISQ = None

    def assess_input(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.IRET), np.integer) == True , \
            'IRET must be int'
        assert self.IRET == 0 , \
            'IRET must be =0 for now'

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NITER), np.integer) == True , \
            'NITER must be int'
            
        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NCORES), np.integer) == True , \
            'NCORES must be int'
        assert self.NCORES >= 1 , \
            'NCORES must be >= 1'

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.PHILIMIT), np.float64) == True , \
            'IRET must be int'
        assert self.PHILIMIT > 0 , \
            'PHILIMIT must be >0'

    def write_input_hdf5(self,runname):
        """
        Write the Retrieval properties into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess_input()

        f = h5py.File(runname+'.h5','a')
        #Checking if Retrieval already exists
        if ('/Retrieval' in f)==True:
            del f['Retrieval']   #Deleting the Atmosphere information that was previously written in the file

        grp = f.create_group("Retrieval")

        dset = grp.create_dataset('NITER',data=self.NITER)
        dset.attrs['title'] = "Maximum number of iterations"
        
        dset = grp.create_dataset('NCORES',data=self.NCORES)
        dset.attrs['title'] = "Number of cores available for parallel computations"

        dset = grp.create_dataset('PHILIMIT',data=self.PHILIMIT)
        dset.attrs['title'] = "Percentage convergence limit"
        dset.attrs['units'] = "%"

        dset = grp.create_dataset('IRET',data=self.IRET)
        dset.attrs['title'] = "Retrieval engine type"
        if self.IRET==0:
            dset.attrs['type'] = "Optimal Estimation"

        f.close()

    def write_output_hdf5(self,runname,Variables,write_cov=True):
        """
        Write the Retrieval outputs into an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','a')
        
        #Checking if Retrieval already exists
        if ('/Retrieval' in f)==True:
            del f['Retrieval']   #Deleting the Atmosphere information that was previously written in the file

        grp = f.create_group("Retrieval")

        dset = grp.create_dataset('NITER',data=self.NITER)
        dset.attrs['title'] = "Maximum number of iterations"

        dset = grp.create_dataset('PHILIMIT',data=self.PHILIMIT)
        dset.attrs['title'] = "Percentage convergence limit"
        dset.attrs['units'] = "%"

        dset = grp.create_dataset('IRET',data=self.IRET)
        dset.attrs['title'] = "Retrieval engine type"
        if self.IRET==0:
            dset.attrs['type'] = "Optimal Estimation"

        #Optimal Estimation
        #####################################################################

        if self.IRET==0:
            
            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/PHI', data=self.PHI)
            dset.attrs['title'] = "'Cost' of retrieved state vector (calculated by cost function, is a balance of fitting modelled spectra and similarity of retrieved state vector to apriori values)."
            
            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/CHISQ', data=self.CHISQ)
            dset.attrs['title'] = "Goodness of fit between modelled spectra and measured spectra"
            
            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/NY', data=self.NY)
            dset.attrs['title'] = "Number of elements in measurement vector"

            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/Y', data=self.Y)
            dset.attrs['title'] = "Measurement vector"

            assert is_diagonal(self.SE), f"Measurement vector covariance matrix must be diagonal"

            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/SE', data=np.sqrt(np.diagonal(self.SE)))
            dset.attrs['title'] = "Uncertainty in Measurement vector (is the square root of the diagonal of the Measurement covariance matrix, which is always a diagonal matrix)"

            dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/YN', data=self.YN)
            dset.attrs['title'] = "Modelled measurement vector"
            
            if write_cov==True:

                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/NX', data=self.NX)
                dset.attrs['title'] = "Number of elements in state vector"

                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/XN', data=self.XN)
                dset.attrs['title'] = "Retrieved state vector"

                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/SX', data=self.ST)
                dset.attrs['title'] = "Retrieved covariance matrix"

                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/XA', data=self.XA)
                dset.attrs['title'] = "A priori state vector"

                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/SA', data=self.SA)
                dset.attrs['title'] = "A priori covariance matrix"
                
                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/KK', data=self.KK)
                dset.attrs['title'] = "Jacobian matrix"
                
                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/AA', data=self.AA)
                dset.attrs['title'] = "Averaging kernel"
                
                dset = h5py_helper.store_data(f, 'Retrieval/Output/OptimalEstimation/DD', data=self.DD)
                dset.attrs['title'] = "Gain matrix"


        #Writing the parameters in the same form as the input .apr file
        APRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        APRERRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        RETPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        RETERRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        ix = 0
        for ivar in range(Variables.NVAR):

            for i in range(Variables.NXVAR[ivar]):
                
                xa1 = self.XA[ix]
                ea1 = np.sqrt(abs(self.SA[ix,ix]))
                xn1 = self.XN[ix] if self.XN is not None else np.nan
                en1 = np.sqrt(abs(self.ST[ix,ix])) if self.ST is not None else np.nan
                if Variables.LX[ix]==1:
                    xa1 = np.exp(xa1)
                    ea1 = xa1*ea1
                    xn1 = np.exp(xn1)
                    en1 = xn1*en1

                RETPARAM[i,ivar] = xn1
                RETERRPARAM[i,ivar] = en1
                APRPARAM[i,ivar] = xa1
                APRERRPARAM[i,ivar] = ea1

                ix = ix + 1

        RETPARAM = None if np.any(np.isnan(RETPARAM)) else RETPARAM
        RETERRPARAM = None if np.any(np.isnan(RETERRPARAM)) else RETERRPARAM

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/NVAR', data=Variables.NVAR)
        dset.attrs['title'] = "Number of retrieved model parameterisations"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/NXVAR', data=Variables.NXVAR)
        dset.attrs['title'] = "Number of parameters associated with each model parameterisation"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/VARIDENT', data=Variables.VARIDENT)
        dset.attrs['title'] = "Variable parameterisation ID"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/VARPARAM', data=Variables.VARPARAM)
        dset.attrs['title'] = "Extra parameters required to model the parameterisations (not retrieved)"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/RETPARAM', data=RETPARAM)
        dset.attrs['title'] = "Retrieved parameters required to model the parameterisations"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/RETERRPARAM', data=RETERRPARAM)
        dset.attrs['title'] = "Uncertainty in the retrieved parameters required to model the parameterisations"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/APRPARAM', data=APRPARAM)
        dset.attrs['title'] = "A priori parameters required to model the parameterisations"

        dset = h5py_helper.store_data(f, 'Retrieval/Output/Parameters/APRERRPARAM', data=APRERRPARAM)
        dset.attrs['title'] = "Uncertainty in the a priori parameters required to model the parameterisations"

        f.close()

    def read_hdf5(self,runname):
        """
        Read the Retrieval properties from an HDF5 file
        """

        import h5py
        from archnemesis.helpers import h5py_helper
        

        with h5py.File(runname+'.h5','r') as f:

            #Checking if Surface exists
            e = "/Retrieval" in f
            if e==False:
                raise ValueError('error :: Retrieval is not defined in HDF5 file')
            else:

                self.NITER = h5py_helper.retrieve_data(f, 'Retrieval/NITER', np.int32)
                self.IRET = h5py_helper.retrieve_data(f, 'Retrieval/IRET', np.int32)
                self.PHILIMIT = h5py_helper.retrieve_data(f, 'Retrieval/PHILIMIT', np.float64)

                #Checking if Retrieval already exists
                if ('/Retrieval/Output' in f)==True:
                    self.PHI = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/PHI', np.float64)
                    self.CHISQ = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/CHISQ', np.float64)

                    self.NX = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/NX', np.int32)
                    self.NY = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/NY', np.int32)

                    self.XN = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/XN', np.array)
                    self.XA = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/XA', np.array)
                    self.ST = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/SX', np.array)
                    self.SA = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/SA', np.array)
                    self.KK = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/KK', np.array)
                    self.AA = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/AA', np.array)
                    self.DD = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/DD', np.array)
                    
                    self.YN = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/YN', np.array)
                    self.Y = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/Y', np.array)
                    y_error = h5py_helper.retrieve_data(f, 'Retrieval/Output/OptimalEstimation/SE', np.array)
                    self.SE = np.diagflat(y_error**2) if y_error is not None else None


    def edit_KK(self, KK_array):
        """
        Edit the Jacobian Matrix
        @param KK_array: 2D array
            Jacobian matrix
        """
        KK_array = np.array(KK_array)
        assert KK_array.shape == (self.NY, self.NX),\
            'KK should be NY by NX.'

        self.KK = KK_array

    def edit_Y(self, Y_array):
        """
        Edit the measurement vector
        @param Y_array: 1D array
            Measurement vector
        """
        Y_array = np.array(Y_array)
        assert len(Y_array) == (self.NY),\
            'Y should be NY.'

        self.Y = Y_array

    def edit_YN(self, YN_array):
        """
        Edit the modelled measurement vector
        @param YN_array: 1D array
            Modelled measurement vector
        """
        YN_array = np.array(YN_array)
        assert len(YN_array) == (self.NY),\
            'YN should be NY.'

        self.YN = YN_array

    def edit_SE(self, SE_array):
        """
        Edit the Measurement covariance matrix
        @param SE_array: 2D array
            Measurement covariance matrix
        """
        SE_array = np.array(SE_array)
        assert SE_array.shape == (self.NY, self.NY),\
            'SE should be NY by NY.'
        self.SE = SE_array

    def edit_XN(self, XN_array):
        """
        Edit the current state vector
        @param XN_array: 1D array
            State vector
        """
        XN_array = np.array(XN_array)
        assert len(XN_array) == (self.NX),\
            'XN should be NX.'
        self.XN = XN_array

    def edit_XA(self, XA_array):
        """
        Edit the a priori state vector
        @param XA_array: 1D array
            A priori State vector
        """
        XA_array = np.array(XA_array)
        assert len(XA_array) == (self.NX),\
            'XA should be NX.'
        self.XA = XA_array

    def edit_SA(self, SA_array):
        """
        Edit the A priori covariance matrix
        @param SA_array: 2D array
            A priori covariance matrix
        """
        SA_array = np.array(SA_array)
        assert SA_array.shape == (self.NX, self.NX),\
            'SA should be NX by NX.'
        self.SA = SA_array
    
    def calc_gain_matrix(self):
        """
        Calculate gain matrix and averaging kernels. The gain matrix is calculated with
            dd = sx*kk_T*(kk*sx*kk_T + se)^-1    (if nx>=ny)
            dd = ((sx^-1 + kk_T*se^-1*kk)^-1)*kk_T*se^-1  (if ny>nx)
        """

        # Calculating the transpose of kk
        kt = self.KK.T

        # Calculating the gain matrix dd
        if self.NX == self.NY:
            # Calculate kk*sa*kt
            a = self.KK @ (self.SA @ kt) + self.SE

            # Inverting a
            c = np.linalg.inv(a)

            # Multiplying (sa*kt) by c
            self.DD = (self.SA @ kt) @ c

        else:
            # Calculating the inverse of Sa and Se
            sai = np.linalg.inv(self.SA)
            sei_inv = np.diag(1.0 / np.diag(self.SE))

            # Calculate kt*sei_inv*kk
            a = kt @ sei_inv @ self.KK + sai

            # Invert a
            c = np.linalg.inv(a)

            # Multiplying c by (kt*sei_inv)
            self.DD = c @ (kt @ sei_inv)

        self.AA = self.DD @ self.KK

    def calc_phiret(self):
        """
        Calculate the retrieval cost function to be minimized in the optimal estimation
        framework, which combines departure from a priori and closeness to spectrum.
        """
        # Calculate values for later
        
        # Use this to minimise difference in `y`
        b = self.YN[:self.NY] - self.Y[:self.NY]
        d = self.XN[:self.NX] - self.XA[:self.NX]
        sai = np.linalg.inv(self.SA)
        sei_inv = np.diag(1.0 / np.diag(self.SE))
        
        ## Getting (yn-y)^2/sigma_y^2 ##
        

        # Multiplying se_inv*b
        a = sei_inv @ b

        # Multiplying bt*a so that (yn-y)^T * se_inv * (yn-y)
        measurement_diff_cost = b.T @ a

        self.CHISQ = measurement_diff_cost / self.NY
        

        ## Getting (xn - x)^2/sigma_x^2 ##

        # Multiply sa_inv*d
        e = sai @ d

        # Multiply dt*e so that (xn-xa)^T * sa_inv * (xn-xa)
        apriori_diff_cost = d.T @ e

        _lgr.warning(f'calc_phiret: {measurement_diff_cost=}, {apriori_diff_cost=}, chisq={self.CHISQ}, {measurement_diff_cost+apriori_diff_cost=}')
        self.PHI = measurement_diff_cost + apriori_diff_cost
        
        assert not np.isnan(self.PHI), "PHI cannot be NAN"
        assert not np.isnan(self.CHISQ), "CHISQ cannot be NAN"
        

    def assess(self):
        """
        This subroutine assesses the retrieval matrices to see whether an exact retrieval may be expected.
        """

        #Calculating transpose of kk
        kt = np.transpose(self.KK)

        #Multiply sa*kt
        m = np.matmul(self.SA,kt)

        #Multiply kk*m so that a = kk*sa*kt
        a = np.matmul(self.KK,m)

        #Add se to a
        b = np.add(a,self.SE)

        #sum1 = 0.0
        #sum2 = 0.0
        #sum3 = 0.0
        #for i in range(self.NY):
        #    sum1 = sum1 + b[i,i]
        #    sum2 = sum2 + self.SE[i,i]
        #    sum3 = sum3 + b[i,i]/self.SE[i,i]

        sum1 = np.sum(np.diagonal(b))
        sum2 = np.sum(np.diagonal(self.SE))
        sum3 = np.sum(np.diagonal(b)/np.diagonal(self.SE))

        sum1 = sum1/self.NY
        sum2 = sum2/self.NY
        sum3 = sum3/self.NY
  
        _lgr.info('Assess:')
        _lgr.info('Average of diagonal elements of Kk*Sx*Kt : '+str(sum1))
        _lgr.info('Average of diagonal elements of Se : '+str(sum2))
        _lgr.info('Ratio = '+str(sum1/sum2))
        _lgr.info('Average of Kk*Sx*Kt/Se element ratio : '+str(sum3))
        if sum3 > 10.0:
            _lgr.info('******************* ASSESS WARNING *****************')
            _lgr.info('Insufficient constraint. Solution likely to be exact')
            _lgr.info('****************************************************')

    def calc_next_xn(self):
        """
        This subroutine performs the optimal estimation retrieval of the
        vector x from a set of measurements y and forward derivative matrix
        kk. The equation solved is (re: p147 of Houghton, Taylor and Rodgers):

                    xn+1 = x0 + dd*(y-yn) - aa*(x0 - xn)    
        """

        m1 = np.zeros([self.NY,1])
        m1[:,0] = self.Y - self.YN
        #dd1 = np.zeros([self.NX,self.NY])
        #dd1[0:nx,0:ny] = dd[0:nx,0:ny]

        m2 = np.zeros([self.NX,1])
        m2[:,0] = self.XA - self.XN
        #aa1 = np.zeros([nx,nx])
        #aa1[0:nx,0:nx] = aa[0:nx,0:nx]

        mp1 = np.matmul(self.DD,m1)
        mp2 = np.matmul(self.AA,m2)

        x_out = np.zeros(self.NX)

        #for i in range(self.NX):
        #    x_out[i] = self.XA[i] + mp1[i,0] - mp2[i,0]
        x_out = self.XA + mp1[:self.NX,0] - mp2[:self.NX,0]
        
        return x_out

    def calc_serr(self):
        """
         Calculates the error covariance matrices after the final iteration has been completed.

        The subroutine calculates the MEASUREMENT ERROR covariance matrix according to the 
        equation (re: p130 of Houghton, Taylor and Rodgers) :
               
                                  sm = dd*se*dd_T

        The subroutine calculates the SMOOTHING ERROR covariance matrix according to the equation:
  
                                  sn = (aa-I)*sx*(aa-I)_T  

        The subroutine also calculates the TOTAL ERROR matrix:

                                  st=sn+sm
        """

        #Multiplying dd*se
        a = np.matmul(self.DD,self.SE)

        #Multiplying a*dt so that dd*se*dt
        dt = np.transpose(self.DD)
        self.SM = np.matmul(a,dt)

        #Calculate aa-ii where I is a diagonal matrix
        b = deepcopy(self.AA)
        for i in range(self.NX):
            b[i,i] = b[i,i] - 1.0
        bt = np.transpose(b)

        #Multiply b*sa so that (aa-I)*sa
        c = np.matmul(b,self.SA)
  
        #Multiply c*bt so tthat (aa-I)*sx*(aa-I)_T  
        self.SN = np.matmul(c,bt)

        #Add sn and sm and get total retrieved error
        self.ST = np.add(self.SN,self.SM)

    def write_mre(self,runname,Variables,Measurement):
        """
        Write the results of an Optimal Estimation retrieval into the .mre file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        @param Measurement: class
            Python class descrbing the measurement and observation
        """

        #Opening file
        f = open(runname+'.mre','w')
    
        str1 = '! Total number of retrievals'
        nspec = 1
        f.write("\t" + str(nspec)+ "\t" + str1 + "\n")

        for ispec in range(nspec):
 
            #Writing first lines
            ispec1 = ispec + 1
            str2 = '! ispec,ngeom,ny,nx,ny'
            f.write("\t %i %i %i %i %i \t %s \n" % (ispec,Measurement.NGEOM,self.NY,self.NX,self.NY,str2)) 
            str3 = 'Latitude, Longitude'
            f.write("\t %5.7f \t %5.7f \t %s \n" % (Measurement.LATITUDE,Measurement.LONGITUDE,str3)) 

            if Measurement.ISPACE==WaveUnit.Wavenumber_cm: #Wavenumber space (cm-1)
                if Measurement.IFORM==SpectraUnit.Radiance: #0
                    str4='Radiances expressed as nW cm-2 sr-1 (cm-1)-1'       
                    xfac=1.0e9
                elif Measurement.IFORM==SpectraUnit.FluxRatio: #1
                    str4='F_plan/F_star Ratio of planet'
                    xfac = 1.0
                elif Measurement.IFORM==SpectraUnit.A_Ratio: #2
                    str4='Transit depth: 100*Planet_area/Stellar_area'
                    xfac = 1.0
                elif Measurement.IFORM==SpectraUnit.Integrated_spectral_power: #3
                    str4='Spectral Radiation of planet: W (cm-1)-1'
                    xfac=1.0e18
                elif Measurement.IFORM==SpectraUnit.Atmospheric_transmission: #4
                    str4='Solar flux: W cm-2 (cm-1)-1'
                    xfac=1.0
                elif Measurement.IFORM==SpectraUnit.Normalised_radiance: #5
                    str4='Transmission'
                    xfac=1.0
                else:
                    _lgr.warning(' in .mre :: IFORM not defined. Default=0')
                    str4='Radiances expressed as nW cm-2 sr-1 cm' 
                    xfac=1.0e9

            elif Measurement.ISPACE==WaveUnit.Wavelength_um: #Wavelength space (um)

                if Measurement.IFORM==SpectraUnit.Radiance: #0
                    str4='Radiances expressed as uW cm-2 sr-1 um-1'       
                    xfac=1.0e6
                elif Measurement.IFORM==SpectraUnit.FluxRatio: #1
                    str4='F_plan/F_star Ratio of planet'
                    xfac = 1.0
                elif Measurement.IFORM==SpectraUnit.A_Ratio: #2
                    str4='Transit depth: 100*Planet_area/Stellar_area'
                    xfac = 1.0
                elif Measurement.IFORM==SpectraUnit.Integrated_spectral_power: #3
                    str4='Spectral Radiation of planet: W um-1'
                    xfac=1.0e18
                elif Measurement.IFORM==SpectraUnit.Atmospheric_transmission: #4
                    str4='Solar flux: W cm-2 um-1'
                    xfac=1.0
                elif Measurement.IFORM==SpectraUnit.Normalised_radiance: #5
                    str4='Transmission'
                    xfac=1.0
                else:
                    _lgr.warning(' in .mre :: IFORM not defined. Default=0')
                    str4='Radiances expressed as uW cm-2 sr-1 um-1' 
                    xfac=1.0e6

            f.write(str4+"\n")

            #Writing spectra
            l = ['i','lambda','R_meas','error','%err','R_fit','%Diff']
            f.write("\t %s %s %s %s %s %s %s \n" % (l[0],l[1],l[2],l[3],l[4],l[5],l[6]))
            ioff = 0
            for igeom in range(Measurement.NGEOM):
                for iconv in range(Measurement.NCONV[igeom]):
                    i = ioff+iconv
                    err1 = np.sqrt(self.SE[i,i])
                    if self.Y[i] != 0.0:
                        xerr1 = abs(100.0*err1/self.Y[i])
                        relerr = abs(100.0*(self.Y[i]-self.YN[i])/self.Y[i])
                    else:
                        xerr1=-1.0
                        relerr1=-1.0

                    if Measurement.IFORM==SpectraUnit.Radiance: #0
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    elif Measurement.IFORM==SpectraUnit.FluxRatio: #1
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    elif Measurement.IFORM==SpectraUnit.A_Ratio: #2
                        strspec = "\t %4i %9.4f %12.6e %12.6e %6.2f %12.6e %6.2f \n"
                    elif Measurement.IFORM==SpectraUnit.Integrated_spectral_power: #3
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    elif Measurement.IFORM in (SpectraUnit.Atmospheric_transmission, SpectraUnit.Normalised_radiance): #4, 5
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8e %9.5f \n"

                    f.write(strspec % (i+1,Measurement.VCONV[iconv,igeom],self.Y[i]*xfac,err1*xfac,xerr1,self.YN[i]*xfac,relerr))
                
                ioff = ioff + Measurement.NCONV[igeom]     

            #Writing a priori and retrieved state vectors
            str1 = '' 
            f.write(str1+"\n")
            f.write('nvar=    '+str(Variables.NVAR)+"\n")
            
            nxtemp = 0
            for ivar in range(Variables.NVAR):

                f.write('Variable '+str(ivar+1)+"\n")
                f.write("\t %i \t %i \t %i\n" % (Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2]))
                f.write("%10.8e \t %10.8e \t %10.8e \t %10.8e \t %10.8e\n" % (Variables.VARPARAM[ivar,0],Variables.VARPARAM[ivar,1],Variables.VARPARAM[ivar,2],Variables.VARPARAM[ivar,3],Variables.VARPARAM[ivar,4]))

                l = ['i','ix','xa','sa_err','xn','xn_err']
                f.write("\t %s %s %s %s %s %s\n" % (l[0],l[1],l[2],l[3],l[4],l[5]))
                for ip in range(Variables.NXVAR[ivar]):
                    ix = nxtemp + ip 
                    xa1 = self.XA[ix]
                    ea1 = np.sqrt(abs(self.SA[ix,ix]))
                    xn1 = self.XN[ix]
                    en1 = np.sqrt(abs(self.ST[ix,ix]))
                    if Variables.LX[ix]==1:
                        xa1 = np.exp(xa1)
                        ea1 = xa1*ea1
                        xn1 = np.exp(xn1)
                        en1 = xn1*en1
                    
                    strx = "\t %4i %4i %12.5e %12.5e %12.5e %12.5e \n"
                    f.write(strx % (ip+1,ix+1,xa1,ea1,xn1,en1))

                nxtemp = nxtemp + Variables.NXVAR[ivar]  

        f.close()  

    def write_cov(self,runname,Variables,pickle=False):
        """
        Write information about the Optimal Estimation matrices into the .cov file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        """

        if pickle==False:
            #Open file
            f = open(runname+'.cov','w')

            npro=1
            f.write("%i %i\n" % (npro,Variables.NVAR))

            for ivar in range(Variables.NVAR):
                f.write("%i \t %i \t %i\n" % (Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2]))
                f.write("%10.8e \t %10.8e \t %10.8e \t %10.8e \t %10.8e\n" % (Variables.VARPARAM[ivar,0],Variables.VARPARAM[ivar,1],Variables.VARPARAM[ivar,2],Variables.VARPARAM[ivar,3],Variables.VARPARAM[ivar,4]))

            f.write("%i %i\n" % (self.NX,self.NY))

            for i in range(self.NX):
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.SA[i,j]))
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.SM[i,j]))
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.SN[i,j]))
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.ST[i,j]))

            for i in range(self.NX):
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.AA[i,j]))

            for i in range(self.NX):
                for j in range(self.NY):
                    f.write("%10.8e\n" % (self.DD[i,j]))

            for i in range(self.NY):
                for j in range(self.NX):
                    f.write("%10.8e\n" % (self.KK[i,j]))

            for i in range(self.NY):
                f.write("%10.8e\n" % (self.SE[i,i]))

            f.close() 

        else:

            import pickle
            filehandler = open(runname+'.cov',"wb")
            pickle.dump(self,filehandler,pickle.HIGHEST_PROTOCOL)

    def read_cov(self,runname,Variables=None,pickle=False):
        """
        Write information about the Optimal Estimation matrices into the .cov file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        """

        if pickle==False:
            if Variables==None:
                Variables=Variables_0()
        
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

            Variables.NVAR = nvar
            Variables.VARIDENT = varident
            Variables.VARPARAM = varparam
            Variables.calc_NXVAR(npro)

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

            self.NX = nx
            self.NY = ny
            self.edit_SA(sa)
            self.edit_SE(se)
            self.SM = sm
            self.SN = sn
            self.ST = st
            self.DD = dd
            self.AA = aa
            self.edit_KK(kk)

        else:

            import pickle

            filen = open(runname+'.cov','rb')
            pickleobj = pickle.load(filen)
            self.NX = pickleobj.NX
            self.NY = pickleobj.NY
            self.SA = pickleobj.SA
            self.SE = pickleobj.SE
            self.SM = pickleobj.SM
            self.SN = pickleobj.SN
            self.ST = pickleobj.ST
            self.DD = pickleobj.DD
            self.AA = pickleobj.AA
            self.KK = pickleobj.KK

    def plot_K(self):
        """
        Function to plot the Jaxobian matrix
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        ax1.set_title('Jacobian Matrix')
        # Center limits around zero
        vmin = np.nanmin(self.KK)
        vmax = np.nanmax(self.KK)
        centered_limit = max(abs(vmin),abs(vmax))
        
        im = ax1.imshow(np.transpose(self.KK),aspect='auto',origin='lower',cmap='jet', vmin=-centered_limit, vmax=centered_limit)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Gradients (dR/dx)')
        ax1.set_ylabel('state vector element #')
        ax1.set_xlabel('measurement vector element #')
        ax1.grid()
        plt.tight_layout()
        plt.show()

    def plot_bestfit(self):
        """
        Function to plot the comparison between modelled and measured spectra
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        fig = plt.figure(figsize=(10,4))
        ax1 = plt.subplot2grid((2,3),(0,0),rowspan=1,colspan=2)
        ax1.set_title('Measured and modelled spectra')
        ax2 = plt.subplot2grid((2,3),(1,0),rowspan=1,colspan=2)
        ax2.set_title('Spectra residual')

        ax3 = plt.subplot2grid((2,3),(0,2),rowspan=1,colspan=1)
        ax3.set_title('apriori and retrieved state vector')
        ax4 = plt.subplot2grid((2,3),(1,2),rowspan=1,colspan=1)
        ax4.set_title('state vector residual')
        
        ax1.plot(range(self.NY),self.Y,c='black',label='Measured spectra', alpha=0.6)
        ax1.plot(range(self.NY),self.YN,c='tab:red',label='Modelled spectra', alpha=0.6)
        ax1.set_xlabel('Measurement vector element #')
        ax1.set_ylabel('Radiance')
        ax1.grid()
        ax1.legend()
        
        ax2.plot(range(self.NY),self.Y-self.YN,c='tab:red', alpha=0.6)
        ax2.set_xlabel('Measurement vector element #')
        ax2.set_ylabel('Residual Radiance')
        ax2.grid()

        ax3.plot(range(self.NX),self.XA,c='black',label='Apriori state vector', alpha=0.6)
        ax3.plot(range(self.NX),self.XN,c='tab:red',label='Retrieved state vector', alpha=0.6)
        ax3.set_xlabel('State vector element #')
        ax3.set_ylabel('State vector element value')
        ax3.grid()
        ax3.legend()

        ax4.plot(range(self.NX),self.XA-self.XN,c='tab:red', alpha=0.6)
        ax4.set_xlabel('State vector element #')
        ax4.set_ylabel('Residual value of state vector element')
        ax4.grid()
        
        plt.tight_layout()
        plt.show()




###############################################################################################
###############################################################################################
#   OPTIMAL ESTIMATION CONVERGENCE LOOP
###############################################################################################
###############################################################################################

def coreretOE(
        runname,
        Variables,
        Measurement,
        Atmosphere,
        Spectroscopy,
        Scatter,
        Stellar,
        Surface,
        CIA,
        Layer,
        Telluric,
        NITER=10,
        PHILIMIT=0.1,
        NCores=1,
        nemesisSO=False,
        write_itr=True,
        return_forward_model=False,
        return_phi_and_chisq_history=False,
    ) -> (
        OptimalEstimation_0 
        | tuple[OptimalEstimation_0, ForwardModel_0]
        | tuple[OptimalEstimation_0, np.ndarray, np.ndarray]
        | tuple[OptimalEstimation_0, ForwardModel_0, np.ndarray, np.ndarray]
    ):


    """
        FUNCTION NAME : coreretOE()
        
        DESCRIPTION : 

            This subroutine runs the Optimal Estimation iterarive algorithm to solve the inverse
            problem and find the set of parameters that fit the spectral measurements and are closest
            to the a priori estimates of the parameters.

        INPUTS :
       
            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the spectroscopic parameters of gaseous species
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations
            Telluric :: Python class defining the parameters to calculate the telluric absorption

        OPTIONAL INPUTS:

            NITER :: Number of iterations in retrieval
            PHILIMIT :: Percentage convergence limit. If the percentage reduction of the cost function PHI
                        is less than philimit then the retrieval is deemed to have converged.

            nemesisSO :: If True, the retrieval uses the function jacobian_nemesisSO(), adapated specifically
                         for solar occultation observations, rather than the more general jacobian_nemesis() function.
            
            return_forward_model :: if True will return the ForwardModel as well as the OptimalEstimation.

        OUTPUTS :

            OptimalEstimation :: Python class defining all the variables required as input or output
                                 from the Optimal Estimation retrieval
 
        CALLING SEQUENCE:
        
            OptimalEstimation = coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer,Telluric)
 
        MODIFICATION HISTORY : Juan Alday (06/08/2021)

    """

    from archnemesis import OptimalEstimation_0
    from archnemesis import ForwardModel_0

    #Creating class and including inputs
    #############################################

    OptimalEstimation = OptimalEstimation_0()

    OptimalEstimation.NITER = NITER
    OptimalEstimation.PHILIMIT = PHILIMIT
    OptimalEstimation.NCORES = NCores
    OptimalEstimation.NX = Variables.NX
    OptimalEstimation.NY = Measurement.NY
    OptimalEstimation.edit_XA(Variables.XA)
    OptimalEstimation.edit_XN(Variables.XN)
    OptimalEstimation.edit_SA(Variables.SA)
    OptimalEstimation.edit_Y(Measurement.Y)
    OptimalEstimation.edit_SE(Measurement.SE)

    phi_history = np.full((NITER+1,), fill_value=np.nan)
    chisq_history = np.full((NITER+1,), fill_value=np.nan)
    state_vector_history = np.full((NITER+1, OptimalEstimation.NX), fill_value=np.nan)
    
    progress_file = f'progress.txt'
    progress_w_iter = max(4, int(np.ceil(np.log10(NITER))))
    progress_fmt = f'{{:0{progress_w_iter}}} | {{:09.3E}} | {{:09.3E}} | {{}}\n'
    progress_head = ('iter' + ('' if progress_w_iter <= 4 else ' '*(progress_w_iter-4))
        +' | phi      '
        +' | chisq    '
        +' | state vector '
        +'\n'
    )
    

    _lgr.info(f'coreretOE :: Starting OptimalEstimation retrieval with NITER={OptimalEstimation.NITER} PHILIMIT={OptimalEstimation.PHILIMIT} NCORES={OptimalEstimation.NCORES}')

    #Opening .itr file
    #################################################################

    if OptimalEstimation.NITER>0:
        if write_itr==True:
            fitr = open(runname+'.itr','w')
            fitr.write("\t %i \t %i \t %i\n" % (OptimalEstimation.NX,OptimalEstimation.NY,OptimalEstimation.NITER))

    #Calculate the first measurement vector and jacobian matrix
    #################################################################

    ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables,Telluric=Telluric)
    _lgr.info('nemesis :: Calculating Jacobian matrix KK')
    YN,KK = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
    
    OptimalEstimation.edit_YN(YN)
    OptimalEstimation.edit_KK(KK)

    #Calculate gain matrix and average kernels
    #################################################################

    _lgr.info('nemesis :: Calculating gain matrix')
    OptimalEstimation.calc_gain_matrix()

    #Calculate initial value of cost function phi
    #################################################################

    _lgr.info('nemesis :: Calculating cost function')
    OptimalEstimation.calc_phiret()

    OPHI = OptimalEstimation.PHI
    _lgr.info('chisq/ny = '+str(OptimalEstimation.CHISQ))
    
    phi_history[0] = OPHI
    chisq_history[0] = OptimalEstimation.CHISQ
    state_vector_history[0,:] = OptimalEstimation.XN
    
    progress_line = progress_fmt.format(0, OptimalEstimation.PHI, OptimalEstimation.CHISQ, ' '.join((f'{x:09.3E}' for x in OptimalEstimation.XN)))
    _lgr.info(f'\t{progress_head}')
    _lgr.info(f'\t{progress_line}')
            
    with open(progress_file, 'w') as f:
        f.write(progress_head)
        f.write(progress_line)

    #Assessing whether retrieval is going to be OK
    #################################################################

    OptimalEstimation.assess()

    #Run retrieval for each iteration
    #################################################################

    #Initializing some variables
    alambda = 1.0   #Marquardt-Levenberg-type 'braking parameter'
    NX11 = np.zeros(OptimalEstimation.NX)
    XN1 = deepcopy(OptimalEstimation.XN)
    NY1 = np.zeros(OptimalEstimation.NY)
    YN1 = deepcopy(OptimalEstimation.YN)

    for it in range(OptimalEstimation.NITER):

        _lgr.info('nemesis :: Iteration '+str(it)+'/'+str(OptimalEstimation.NITER))

        #Writing into .itr file
        ####################################
        if write_itr==True:
            fitr.write(f'{OptimalEstimation.CHISQ:09.4E} {OptimalEstimation.PHI:09.4E}\n')
            for i in range(OptimalEstimation.NX):fitr.write(f'{XN1[i]:09.4E}\n')#'%10.5f \n' % (XN1[i]))
            for i in range(OptimalEstimation.NX):fitr.write(f'{OptimalEstimation.XA[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.XA[i]))
            for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.Y[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.Y[i]))
            for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.SE[i,i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.SE[i,i]))
            for i in range(OptimalEstimation.NY):fitr.write(f'{YN1[i]:09.4E}\n')#'%10.5f \n' % (YN1[i]))
            for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.YN[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.YN[i]))
            for i in range(OptimalEstimation.NX):
                for j in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.KK[j,i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.KK[j,i]))


        #Calculating next state vector
        #######################################

        _lgr.info('nemesis :: Calculating next iterated state vector')
        X_OUT = OptimalEstimation.calc_next_xn()
        #  x_out(nx) is the next iterated value of xn using classical N-L
        #  optimal estimation. However, we want to apply a braking parameter
        #  alambda to stop the new trial vector xn1 being too far from the
        #  last 'best-fit' value xn

        check_marquardt_brake = True
        while check_marquardt_brake: #We continue in this while loop until we do not find problems with the state vector
            
            if alambda > 1E30:
                raise ValueError('error in nemesis :: Death spiral in braking parameters - stopping')
            
            check_marquardt_brake = False
            
            XN1 = OptimalEstimation.XN + (X_OUT-OptimalEstimation.XN)/(1.0+alambda)
            
            if np.any(((XN1 > 85) | (XN1 < -85)) & (Variables.LX==1)):
                _lgr.info('nemesis :: log(number gone out of range) --- increasing brake')
                alambda *= 10
                check_marquardt_brake = True
            
            else:
                #Check to see if any VMRs or other parameters have gone negative.
                Variables1 = deepcopy(Variables)
                Variables1.XN = XN1

                ForwardModel1 = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Telluric=Telluric,Variables=Variables1)
                ForwardModel1.subprofretg()

                if np.any(ForwardModel1.AtmosphereX.T < 0.0):
                    _lgr.info('nemesis :: Temperature has gone negative --- increasing brake')
                    alambda *= 10
                    check_marquardt_brake = True



        #Calculate test spectrum using trial state vector xn1. 
        #Put output spectrum into temporary spectrum yn1 with
        #temporary kernel matrix kk1. Does it improve the fit? 
        Variables.edit_XN(XN1)
        _lgr.info('nemesis :: Calculating Jacobian matrix KK')

        ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Telluric=Telluric,Variables=Variables)
        YN1,KK1 = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)

        OptimalEstimation1 = deepcopy(OptimalEstimation)
        OptimalEstimation1.edit_YN(YN1)
        OptimalEstimation1.edit_XN(XN1)
        OptimalEstimation1.edit_KK(KK1)
        OptimalEstimation1.calc_phiret()
        _lgr.info('chisq/ny = '+str(OptimalEstimation1.CHISQ))

        #Does the trial solution fit the data better?
        if (OptimalEstimation1.PHI <= OPHI):
            _lgr.info('Successful iteration. Updating xn,yn and kk')
            OptimalEstimation.edit_XN(XN1)
            OptimalEstimation.edit_YN(YN1)
            OptimalEstimation.edit_KK(KK1)
            Variables.edit_XN(XN1) # This seems to be superflous as the same operation happens about 19 lines above

            #Now calculate the gain matrix and averaging kernels
            OptimalEstimation.calc_gain_matrix()

            #Updating the cost function
            OptimalEstimation.calc_phiret()

            #Has the solution converged?
            tphi = 100.0*(OPHI-OptimalEstimation.PHI)/OPHI
            if (tphi>=0.0 and tphi<=OptimalEstimation.PHILIMIT and alambda<1.0):
                _lgr.info('phi, phlimit : '+str(tphi)+','+str(OptimalEstimation.PHILIMIT))
                _lgr.info('Phi has converged')
                _lgr.info('Terminating retrieval')
                break
            else:
                OPHI=OptimalEstimation.PHI
                alambda = alambda*0.3  #reduce Marquardt brake

        else:
            #Leave xn and kk alone and try again with more braking
            alambda *= 10.0  #increase Marquardt brake
        
        phi_history[it+1] = OptimalEstimation.PHI
        chisq_history[it+1] = OptimalEstimation.CHISQ
        state_vector_history[it+1,:] = OptimalEstimation.XN
        
        progress_line = progress_fmt.format(0, OptimalEstimation.PHI, OptimalEstimation.CHISQ, ' '.join((f'{x:09.3E}' for x in OptimalEstimation.XN)))
        _lgr.info(f'\t{progress_head}')
        _lgr.info(f'\t{progress_line}')
                
        with open(progress_file, 'a') as f:
            f.write(progress_line)
    
    _lgr.info(f'coreretOE :: Completed Optimal Estimation retrieval. Showing phi and chisq evolution')
    with open(f'phi_chisq.txt', 'w') as f:
        w_iter = max(4, int(np.ceil(np.log10(NITER))))
        fmt = f'{{:0{w_iter}}} | {{:09.3E}} | {{:09.3E}} | {{}}\n'
        head = ('iter' + ('' if w_iter <= 4 else ' '*(w_iter-4))
            +' | phi      '
            +' | chisq    '
            +' | state vector '
            +'\n'
        )
        _lgr.info(f'\t{head}')
        f.write(head)
        for i,(p,c,sv) in enumerate(zip(phi_history, chisq_history, state_vector_history)):
            line = fmt.format(i, p, c, ' '.join((f'{x:09.3E}' for x in sv)))
            _lgr.info(f'\t{line}')
            f.write(line)
            

    #Writing into .itr file for final iteration
    ####################################
    if write_itr==True:
        fitr.write(f'{OptimalEstimation.CHISQ:09.4E} {OptimalEstimation.PHI:09.4E}\n')
        for i in range(OptimalEstimation.NX):fitr.write(f'{XN1[i]:09.4E}\n')#'%10.5f \n' % (XN1[i]))
        for i in range(OptimalEstimation.NX):fitr.write(f'{OptimalEstimation.XA[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.XA[i]))
        for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.Y[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.Y[i]))
        for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.SE[i,i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.SE[i,i]))
        for i in range(OptimalEstimation.NY):fitr.write(f'{YN1[i]:09.4E}\n')#'%10.5f \n' % (YN1[i]))
        for i in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.YN[i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.YN[i]))
        for i in range(OptimalEstimation.NX):
            for j in range(OptimalEstimation.NY):fitr.write(f'{OptimalEstimation.KK[j,i]:09.4E}\n')#'%10.5f \n' % (OptimalEstimation.KK[j,i]))

    #Calculating output parameters
    ######################################################

    #Calculating retrieved covariance matrices
    OptimalEstimation.calc_serr()

    #Make sure errors stay as a priori for kiter < 0
    if OptimalEstimation.NITER<0:
        OptimalEstimation.ST = deepcopy(OptimalEstimation.SA)

    #Closing .itr file
    if write_itr==True:
        if OptimalEstimation.NITER>0:
            fitr.close()

    #Writing the contribution of each gas to .gcn file
    #if nemesisSO==True:
    #    calc_gascn(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
    
    
    
    result = (OptimalEstimation,)
    
    if return_forward_model:
        result = *result, ForwardModel
        
    if return_phi_and_chisq_history:
        result = *result, phi_history, chisq_history
    
    return result[0] if len(result) == 1 else result
    
