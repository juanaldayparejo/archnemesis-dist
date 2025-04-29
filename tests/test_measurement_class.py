import pytest  
import archnemesis as ans
import numpy as np
  
def test_lblconv():
    '''
    Test convolution of the spectra with the ILS
    '''
    
    # Creating function to be convolved with the ILS 
    ##########################################################################

    dvx = 0.1
    vwave = np.arange(0, 50.+dvx, dvx)
    nwave = len(vwave)

    # For demonstration, let's define y as a Gaussian "peak" near the middle
    true_center = 35.
    true_fwhm   = 1.0
    true_sigma  = 0.5 * true_fwhm / np.sqrt(np.log(2))
    y = np.exp(-((vwave - true_center)/true_sigma)**2)
    

    nshapes = 3
    for ishape in range(nshapes):

        #Performing convolution with archNEMESIS
        ##########################################################################

        Measurement = ans.Measurement_0()
        
        Measurement.NGEOM = 1
        Measurement.FWHM = 1.
        Measurement.ISHAPE = ishape
        Measurement.ISPACE = 0

        vconv = vwave.copy()
        nconv = len(vconv)

        Measurement.NGEOM = 1
        Measurement.NCONV = np.array([len(vconv)])
        vconvx = np.zeros((len(vconv),Measurement.NGEOM))
        vconvx[:,0] = vconv
        Measurement.edit_VCONV(vconvx)

        Measurement.build_ils()  #Calculating the ILS
        yconv_an = Measurement.lblconv(vwave,y,IGEOM=0)
        
        #Performing the convolution with numpy
        ##########################################################################
        
        #Define the same ILS as in archNEMESIS
        vx = Measurement.VFIL[:,0] - Measurement.VCONV[0,0]
        ax = Measurement.AFIL[:,0]

        #Interpolating the ILS to the same grid as the spectrum (required by numpy)
        vwavex = np.arange(vx.min(),vx.max()+dvx,dvx)
        kernel = np.interp(vwavex,vx,ax)
        
        # Normalize kernel area to 1 (required by numpy)
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
            
        yconv_np = np.convolve(y, kernel,mode='same')

        #Comparing the results
        assert np.allclose(yconv_an[np.where(yconv_an/yconv_an.max()>=1.0e-2)], yconv_np[(yconv_an/yconv_an.max()>=1.0e-2)], rtol=3.0e-2)
    
    
