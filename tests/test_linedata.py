import os
import pytest
import numpy as np
import archnemesis as ans
import archnemesis.lineshape

if False:
    print(pytest) # Removes the "unused import" error

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
    LineData = ans.LineData_0(
        ID=5,
        ISO=1,
        LINE_DATABASE="./CO_1_ambient_AIR.h5",  #HITRAN24
        PARTITION_FUNCTION_DATABASE="./CO_1_ambient_AIR.h5"
    )

    LineData.fetch_partition_fn()
    LineData.fetch_linedata()

    # Point 1
    itemp = 5 ; ipress = 5
    temp = Spectroscopy.TEMP[itemp] ; press = Spectroscopy.PRESS[ipress]
    expected = Spectroscopy.K[:,ipress,itemp,0]

    k = LineData.add_monochromatic_absorption(
            wave_grid = Spectroscopy.WAVE,
            t_calc = temp,
            p_calc = press,
            amb_frac = 1.0,
            wave_unit = Spectroscopy.ISPACE,
            lineshape_fn = ans.lineshape.voigt,
            wn_calc_window = 25.0, # (cm^{-1})
            wn_approx_window = 25.0, # (cm^{-1})
            include_pressure_shift = False
    )
    
    atol = np.quantile(expected, 0.25) * 1E-1 # Numpy default value is 1E-8 which is too large for small values
    rtol = 0.4
    
    if False: # Extra diagnostics. To use, just set `if False:` to `if True:`
        import matplotlib.pyplot as plt
        
        tol = (atol + rtol*expected)
        expected_tol_plus = expected + tol
        expected_tol_minus = expected - tol
        residual = k - expected
        residual_pos = residual[residual>0]
        residual_neg = residual[residual<0]
        
        fig1 = plt.figure()
        plt.fill_between(Spectroscopy.WAVE, expected_tol_minus, expected_tol_plus, alpha=0.1, color='blue', label='Valid expected region')
        plt.plot(Spectroscopy.WAVE, expected, alpha=0.6, color='tab:blue', ls='--', label='expected')
        plt.plot(Spectroscopy.WAVE, k, alpha=0.6, color='tab:orange', ls=':', label='calculated {k}')
        plt.legend()
        plt.title('Calculated Absorption Coefficient {k} and Expected Absorption Coeffiient {expected}')
        plt.xlabel('Wavenumber (cm^{-1})')
        plt.ylabel('LOG Absorption Coefficient (1 / [molec cm^{-2}])')
        plt.yscale('log')
        
        fig2 = plt.figure()
        plt.fill_between(Spectroscopy.WAVE, -tol, tol, alpha=0.1, color='blue', label='Valid expected region')
        plt.plot(Spectroscopy.WAVE, residual, alpha=0.6, color='tab:red', ls='-', label='residual {k-expected}')
        plt.legend()
        plt.title('Residual {k - expected}')
        plt.xlabel('Wavenumber (cm^{-1})')
        plt.ylabel('Residual Absorption Coefficient (1 / [molec cm^{-2}])')
        
        fig3 = plt.figure()
        plt.fill_between(Spectroscopy.WAVE, 0, tol, alpha=0.1, color='blue', label='Valid expected region')
        plt.plot(Spectroscopy.WAVE[residual>0], residual_pos, alpha=1, color='tab:red', ls='none', marker='.', markersize=2, markeredgecolor='none', label='positive residual {k-expected}')
        plt.plot(Spectroscopy.WAVE[residual<0], -1*residual_neg, alpha=1, color='tab:purple', ls='none', marker='.', markersize=2, markeredgecolor='none', label='negative residual {k-expected}')
        plt.legend()
        plt.title('Positive and Negative Residual {k - expected}')
        plt.xlabel('Wavenumber (cm^{-1})')
        plt.ylabel('Magnitude of Residual Absorption Coefficient (1 / [molec cm^{-2}])')
        plt.yscale('log')
        
        fig4 = plt.figure()
        plt.fill_between(Spectroscopy.WAVE, -tol/expected, +tol/expected, alpha=0.1, color='blue', label='Valid expected region')
        plt.plot(Spectroscopy.WAVE, residual / expected, alpha=0.6, color='tab:red', ls='-', label='fractional residual {(k-expected)/expected}')
        plt.legend()
        plt.title('Fractional Residual {(k - expected)/expected}')
        plt.xlabel('Wavenumber (cm^{-1})')
        plt.ylabel('Fractional Residual of Absorption Coefficient (RATIO)')
        
        plt.show()
        fig1.savefig('comparison.png', dpi=300, bbox_inches='tight')
        fig2.savefig('residual.png', dpi=300, bbox_inches='tight')
        fig3.savefig('residual_magnitude.png', dpi=300, bbox_inches='tight')
        fig4.savefig('fractional_residual.png', dpi=300, bbox_inches='tight')

    assert np.allclose(k, expected, atol=atol, rtol=rtol)
    