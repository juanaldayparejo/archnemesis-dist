#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from __future__ import annotations #  for 3.9 compatability
import numpy as np


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

"""
Object to calculate the atmospheric paths
"""
class Path_0:
    
    def __init__(self,AtmClass_List,COMBINE=False):
        """
        After performing each of the independent atmospheric calculations included in the .pat file,
        these are merged into the same atmospheric class.

        Inputs
        ------
        @param AtmClass_List: list of class objects
            A list of class objects defining each of the calculations to be performed (AtmCalc_0 classes)
        @param COMBINE: bool
            Flag indicating whether the different calculations can be merged and performed simultaneously
            or need to be performed separately

        Attributes
        -----------
        @attribute NCALC: int
            Number of atmospheric calculations to be performed
        @attribute NPATH: int
            Total number of paths to perform all the NCALC atmospheric calciulations
        @attribute NLAYIN: 1D array
            For each path, number of layers involved in the calculation
        @attribute LAYINC: 2D array
            For each path, layers involved in the calculation
        @attribute EMTEMP: 1D array
            For each path, emission temperature of the layers involved in the calculation
        @attribute SCALE: 1D array
            For each path, scaling factor to calculate line-of-sight density in each layer with respect
            to the vertical line-of-sight density
        @attribute IMOD: 1D array
            For each path, calculation type  
     
        Methods
        -------

        """

        self.COMBINE = COMBINE

        self.NCALC = None
        self.NPATH = None
        self.NPATH1 = None
        self.IMOD = None
        self.NLAYIN = None
        self.LAYINC = None
        self.EMTEMP = None
        self.SCALE = None
        self.ITYPE = None
        self.NINTP = None
        self.ICALD = None
        self.NREALP = None
        self.RCALD = None        

        #Reading number of calculations and estimating total number of paths
        if self.COMBINE==True:
            self.NCALC = 1
        else:
            self.NCALC = len(AtmClass_List)

        NCALC1 = len(AtmClass_List)
        NPATH1 = np.zeros(NCALC1,dtype='int32')
        NPATH = 0
        MLAYIN = 0
        MINTP = 0
        MREALP = 0
        for ICALC in range(NCALC1):
            NPATH1[ICALC] = AtmClass_List[ICALC].NPATH
            NPATH = AtmClass_List[ICALC].NPATH + NPATH
            MLAYIN1 = AtmClass_List[ICALC].NLAYIN.max()
            MINTP1 = AtmClass_List[ICALC].NINTP
            MREALP1 = AtmClass_List[ICALC].NREALP
            if MLAYIN1>MLAYIN:
                MLAYIN = MLAYIN1
            if MINTP1>MINTP:
                MINTP = MINTP1
            if MREALP1>MREALP:
                MREALP = MREALP1

        self.NPATH = NPATH
        if COMBINE==True:
            self.NPATH1 = [NPATH]

        #Creating arrays to store the information
        IMOD = np.zeros(self.NPATH,dtype='int32')
        NLAYIN = np.zeros(self.NPATH,dtype='int32')
        EMISS_ANG = np.zeros(self.NPATH)
        AZI_ANG = np.zeros(self.NPATH)
        SOL_ANG = np.zeros(self.NPATH)
        LAYINC = np.zeros([MLAYIN,self.NPATH],dtype='int32')
        EMTEMP = np.zeros([MLAYIN,self.NPATH])
        SCALE = np.zeros([MLAYIN,self.NPATH])
        ITYPE = np.zeros(self.NCALC,dtype='int32')
        NINTP = np.zeros(self.NCALC,dtype='int32')
        ICALD = np.zeros([3,self.NCALC],dtype='int32')
        NREALP = np.zeros(self.NCALC,dtype='int32')
        RCALD = np.zeros([3,self.NCALC])

        IPATH = 0
        for ICALC in range(NCALC1):

            NPATH1 = AtmClass_List[ICALC].NPATH

            IMOD[IPATH:IPATH+NPATH1] = AtmClass_List[ICALC].IMOD            
            NLAYIN[IPATH:IPATH+NPATH1] = AtmClass_List[ICALC].NLAYIN
            
            EMISS_ANG[IPATH:IPATH+NPATH1] = AtmClass_List[ICALC].EMISS_ANG
            SOL_ANG[IPATH:IPATH+NPATH1] = AtmClass_List[ICALC].SOL_ANG
            AZI_ANG[IPATH:IPATH+NPATH1] = AtmClass_List[ICALC].AZI_ANG

            if self.COMBINE==False:
                ITYPE[ICALC] = AtmClass_List[ICALC].ITYPE
                NINTP[ICALC] = AtmClass_List[ICALC].NINTP
                NREALP[ICALC] = AtmClass_List[ICALC].NREALP
                ICALD[0:NINTP[ICALC],ICALC] = AtmClass_List[ICALC].ICALD[0:NINTP[ICALC]]
                RCALD[0:NREALP[ICALC],ICALC] = AtmClass_List[ICALC].RCALD[0:NREALP[ICALC]]
            else:
                ITYPE[0] = AtmClass_List[ICALC].ITYPE
                NINTP[0] = AtmClass_List[ICALC].NINTP
                NREALP[0] = AtmClass_List[ICALC].NREALP
                ICALD[0:NINTP[0],0] = AtmClass_List[ICALC].ICALD[0:NINTP[0]]
                RCALD[0:NREALP[0],0] = AtmClass_List[ICALC].RCALD[0:NREALP[0]]

            for i in range(NPATH1):
                LAYINC[0:NLAYIN[IPATH+i],IPATH+i] = AtmClass_List[ICALC].LAYINC[0:NLAYIN[i],i]
                SCALE[0:NLAYIN[IPATH+i],IPATH+i] = AtmClass_List[ICALC].SCALE[0:NLAYIN[i],i]
                EMTEMP[0:NLAYIN[IPATH+i],IPATH+i] = AtmClass_List[ICALC].EMTEMP[0:NLAYIN[i],i]

            IPATH = IPATH + NPATH1

        
        self.IMOD = IMOD
        self.NLAYIN = NLAYIN
        self.EMISS_ANG = EMISS_ANG
        self.AZI_ANG = AZI_ANG
        self.SOL_ANG = SOL_ANG
        self.LAYINC = LAYINC
        self.SCALE = SCALE
        self.EMTEMP = EMTEMP
        self.ITYPE = ITYPE
        self.NINTP = NINTP
        self.ICALD = ICALD
        self.NREALP = NREALP
        self.RCALD = RCALD