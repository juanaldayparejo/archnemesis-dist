from __future__ import annotations #  for 3.9 compatability

import numpy as np

import matplotlib.pyplot as plt


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

def is_diagonal(a : np.ndarray) -> bool:
    """
    Tests if array `a` is diagonal or not.
    
    ## ARGUMENTS ##
        a : np.ndarray
            The array to test for diagonalness

    ## RETURNS ##
        is_diagonal : bool
            True if `a` is diagonal, false otherwise
    """
    
    # This is a bit of a hack, but should be pretty fast
    diag_elements : np.ndarray = np.diagonal(a).copy()
    
    try:
        np.fill_diagonal(a, 0)
        result : bool = np.all(a == 0)
    finally:
        np.fill_diagonal(a, diag_elements)
    
    return result


def ngauss(npx,x,ng,iamp,imean,ifwhm,MakePlot=False):


    """
        FUNCTION NAME : ngauss()
        
        DESCRIPTION : 

            Create a function which is the sum of multiple gaussians
 
        INPUTS :
      
            npx :: Number of points in x-array
            x(npx) :: Array specifying the points at which the function must be calculated
            ng :: Number of gaussians
            iamp(ng) :: Amplitude of each of the gaussians
            imean(ng) :: Center x-point of the gaussians
            ifwhm(ng) :: FWHM of the gaussians

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            fun(npx) :: Function at each x-point

        CALLING SEQUENCE:
        
            fun = ngauss(npx,x,ng,iamp,imean,ifwhm)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    fun  = np.zeros([npx])
    isigma = ifwhm/(2.0*np.sqrt(2.*np.log(2.)))
    for i in range(npx):
        for j in range(ng):
            fun[i] = fun[i] + iamp[j] * np.exp( -(x[i]-imean[j])**2.0/(2.0*isigma[j]**2.)  )


    #Make plot if keyword is specified
    if MakePlot == True:
        axis_font = {'size':'20'}
        cm = plt.cm.get_cmap('RdYlBu')
        fig = plt.figure(figsize=(15,8))
        wavemin = x.min()
        wavemax = x.max()
        ax = plt.axes()
        ax.set_xlim(wavemin,wavemax)
        ax.tick_params(labelsize=20)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('x',**axis_font)
        plt.ylabel('f(x)',**axis_font)
        im = ax.plot(x,fun)
        plt.grid()
        plt.show()    
    
    return fun