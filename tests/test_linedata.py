import pytest  
import archnemesis as ans
import numpy as np
import sys,os

def test_lbl_calculation():  
    '''
    Test calculation of line-by-line cross sections

    Right now we are not using the same spectroscopic data, so we allow a large
    relative tolerance (50%) in the comparison. 
    We assume the differences come from the spectroscopy, but this needs to be checked in the future.
    '''

    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/linedata/')
    os.chdir(test_dir) #Changing directory to read files

    #Reading lbl table from NEMESIS (generated with HITRAN20)
    Spectroscopy = ans.Spectroscopy_0()
    Spectroscopy.ILBL = 2
    Spectroscopy.NGAS = 1
    Spectroscopy.ID = [5]
    Spectroscopy.ISO = [1]
    Spectroscopy.LOCATION = ["../../../archnemesis/Data/reference_tables/lbltab_mars/lbltab_co_iso1.lta"]
    Spectroscopy.read_tables()

    #Calculating cross sections with LineData class
    LineData = ans.LineData_1(
        ID=5,
        ISO=1,
        LINE_DATABASE="./CO_1_ambient_AIR.h5"  #HITRAN24
        )

    LineData.fetch_partition_function()

    # Point 1
    itemp = 5 ; ipress = 5
    temp = Spectroscopy.TEMP[itemp] ; press = Spectroscopy.PRESS[ipress]
    expected = Spectroscopy.K[:,ipress,itemp,0]

    k = LineData.calculate_monochromatic_absorption(
            waves = Spectroscopy.WAVE,
            temp = temp,
            press = press,
            amb_frac = 1.,
            wave_unit = Spectroscopy.ISPACE,
            lineshape_fn = ans.Data.lineshapes.voigt,
            line_calculation_wavenumber_window = 25.0,
    )

    assert np.allclose(k, expected, rtol=0.5)
    