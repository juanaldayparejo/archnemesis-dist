#from __future__ import annotations #  for 3.9 compatability

import io
from typing import NamedTuple, TYPE_CHECKING
import struct


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import archnemesis as ans
from archnemesis.database.datatypes.wave_point import WavePoint
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    NPRESS = "Number of points in pressure grid"
    NTEMP_PER_PRESSURE = "Number of temperatures per pressure point, i.e. number of temperature profiles"
    NTEMP = "Number of points in temperature grid"
    NWAVE = "Number of wave points"


_header_struct = struct.Struct('<2i2f4i')

def _get_body_structs(nwave, npress, ntemp):
    _lgr.debug(f'{nwave=} {npress=} {ntemp=} {nwave * npress * abs(ntemp)=}')
    fortran_record_shape = (
        npress, 
        ntemp if ntemp > 0 else -ntemp*npress, 
        nwave * npress * abs(ntemp)
    )
    return (
        fortran_record_shape,
        struct.Struct(f'<{fortran_record_shape[0]}f'),
        struct.Struct(f'<{fortran_record_shape[1]}f'),
        struct.Struct(f'<{fortran_record_shape[2]}f'),
    )

class LblHeader(NamedTuple):
    irec0 : int # This is ignored and always 9
    nwave : int 
    vmin : float
    delv : float
    npress : int
    ntemp : int
    gas_id : int 
    iso_id : int


class LblDataTProfilesAtPressure(NamedTuple):
    """
    Line-by-line data specified at `NPRESS` pressures with `NTEMP_PER_PRESSURE` temperature profiles for each pressure.
    
    Temperature profiles are associated by index, so the 0th temperature across all pressures is the 0th temperature profile.
    """
    gas_id : int # RADTRAN
    iso_id : int # RADTRAN
    wave_unit : ans.enums.WaveUnit # ENUM value
    wave : np.ndarray[['NWAVE'], float] # in `wave_unit`
    press : np.ndarray[['NPRESS'], float] # bar
    temp : np.ndarray[['NPRESS', 'NTEMP_PER_PRESSURE'], float] # kelvin
    k : np.ndarray[['NWAVE','NPRESS','NTEMP_PER_PRESSURE'], float] # 1E20 (cm^2), see function `Spectroscopy_0::write_lbltable` for precident
    
    def write_legacy_header(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy_header(g)
        
        f.write(_header_struct.pack(
            9 + self.press.size + self.temp.size, # "fortran record number" at which absorption coefficient data starts
            self.wave.size,
            self.wave[0],
            self.wave[1] - self.wave[0],
            self.press.size,
            -self.temp.shape[1],
            self.gas_id,
            self.iso_id,
        ))
    
    def write_legacy(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy(g)
        print(f'Writing {self.__class__.__name__} to {f.name}')
        
        self.write_legacy_header(f)
        
        fortran_record_shape, p_struct, t_struct, k_struct = _get_body_structs(self.wave.size, self.press.size, -self.temp.shape[1])
        
        print(f'{p_struct=}')
        f.write(
            p_struct.pack(
                #*(x for x in np.nditer(np.asfortranarray(self.press)))
                *self.press.flat
            )
        )
        
        print(f'{t_struct=}')
        f.write(
            t_struct.pack(
                #*(x for x in np.nditer(np.asfortranarray(self.temp)))
                *self.temp.flat
            )
        )
        
        print(f'{k_struct=}')
        f.write(
            k_struct.pack(
                #*(x for x in np.nditer(np.asfortranarray(self.k))) # already multiplied by 1E20 to avoid underflow when saving single precision floats
                *self.k.flat
            )
        )
        
        _lgr.info(f'Written {self.__class__.__name__} to {f.name}')
        return
    
    def plot(self, wave_unit : None | ans.enums.WaveUnit = None, z_logscale=True):
        """
        Show a map of `k` (color axis) vs pressure (y-axis) and wave (x-axis) for each temperature 
        profile. Also plots the temperature profile as a red-line on a shared y-axis.
        """
        if wave_unit is None:
            wave_unit = self.wave_unit
        
        fig, ax = plt.subplots(self.k.shape[2],1, figsize=(12,6*self.k.shape[2]), squeeze=False)
        ax = ax.flatten()
        ax2 = [a.twiny() for a in ax]
        for a in ax2:
            a.sharex(ax2[0])
        
        
        def get_edges(a):
            e = np.empty((a.size+1,), dtype=a.dtype)
            half_da = (a[1:] - a[:-1])/2
            e[0] = a[0] - half_da[0]
            e[1:-1] = a[:-1]+half_da
            e[-1] = a[-1] + half_da[-1]
            return e
        
        wave_points = WavePoint(self.wave, self.wave_unit).to_unit(wave_unit)
        print(f'{wave_points=}')
        
        p_edges = np.exp(get_edges(np.log(self.press)))
        w_edges = get_edges(wave_points.value)
        
        fig.suptitle(f'Opacities for {RadtranGasDescriptor(self.gas_id, self.iso_id).label} at each temperature profile.')
        
        for i in range(self.k.shape[2]):
            im = ax[i].pcolormesh(
                w_edges,
                p_edges,
                (np.log(self.k[:,:,i].T)) if z_logscale else (self.k[:,:,i].T)
            )
            
            div = make_axes_locatable(ax[i])
            cax = div.append_axes('right', size=0.05, pad=0.0)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            if wave_points.unit == ans.enums.WaveUnit.Wavenumber_cm:
                ax[i].set_xlabel('Wavenumber ($cm^{-1}$)')
                if z_logscale:
                    cax.set_ylabel('log[Absorption Coefficient] log(cm$^{-2}$ / $cm^{-1}$)')
                else:
                    cax.set_ylabel('Absorption Coefficient] (cm$^{-2}$ / $cm^{-1}$)')
            elif wave_points.unit == ans.enums.WaveUnit.Wavelength_um:
                ax[i].set_xlabel(r'Wavelength ($\mu$m)')
                if z_logscale:
                    cax.set_ylabel(r'log[Absorption Coefficient] log(cm$^{-2}$ / $\mu$m)')
                else:
                    cax.set_ylabel(r'Absorption Coefficient (cm$^{-2}$ / $\mu$m)')
            else:
                raise RuntimeError(f'Unknown {wave_unit}. Should be one of {ans.enums.WaveUnit.values()}')
            
            
            
            
            
            ax[i].set_ylabel('Pressure (atm)')
            ax[i].set_yscale('log')
            ax[i].invert_yaxis()
            
            
            ax2[i].plot(self.temp[:,i], self.press, color='red', linestyle='--', alpha=0.6)
            ax2[i].set_xlabel('Temperature (K)')
            
            
            
            
            
        
        plt.show()



class LblDataTPGrid(NamedTuple):
    """
    Line-by-line data specified on a 2-dimensional (`NPRESS`,`NTEMP`) grid. All temperature and pressure
    combinations on the grid have a `k` entry.
    """
    gas_id : int # RADTRAN
    iso_id : int # RADTRAN
    wave_unit : ans.enums.WaveUnit # ENUM value
    wave : np.ndarray[['NWAVE'], float] # wavenumber cm^-1
    press : np.ndarray[['NPRESS'], float] # bar
    temp : np.ndarray[['NTEMP'], float] # Kelvin
    k : np.ndarray[['NWAVE','NPRESS','NTEMP'], float] # 1E20 (cm^2), see function `Spectroscopy_0::write_lbltable` for precident


    def write_legacy_header(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy_header(g)
        
        f.write(_header_struct.pack(
            9 + self.press.size + self.temp.size, # "fortran record number" at which absorption coefficient data starts
            self.wave.size,
            self.wave[0],
            self.wave[1] - self.wave[0],
            self.press.size,
            self.temp.size,
            self.gas_id,
            self.iso_id,
        ))
    
    def write_legacy(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy(g)
        
        self.write_legacy_header(f)
        
        fortran_record_shape, p_struct, t_struct, k_struct = _get_body_structs(self.wave.size, self.press.size, self.temp.size)
        
        f.write(
            p_struct.pack(
                *(x for x in np.nditer(np.asfortranarray(self.press)))
            )
        )
        
        f.write(
            t_struct.pack(
                *(x for x in np.nditer(np.asfortranarray(self.temp)))
            )
        )
        
        f.write(
            k_struct.pack(
                *(x for x in np.nditer(np.asfortranarray(self.k))) # already multiplied by 1E20 to avoid underflow when saving single precision floats
            )
        )

    def plot(self, wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavelength_um):
        """
        Show a map of `k` (color axis) vs pressure (y-axis) and wave (x-axis) for each temperature
        point on the grid.
        """
        fig, ax = plt.subplots(self.k.shape[2],1, figsize=(12,6*self.k.shape[2]), squeeze=False)
        ax = ax.flatten()
        
        wave_points = WavePoint(self.wave, self.wave_unit).to_unit(wave_unit)
        
        def get_edges(a):
            e = np.empty((a.size+1,), dtype=a.dtype)
            half_da = (a[1:] - a[:-1])/2
            e[0] = a[0] - half_da[0]
            e[1:-1] = a[:-1]+half_da
            e[-1] = a[-1] + half_da[-1]
            return e
        
        p_edges = np.exp(get_edges(np.log(self.press)))
        w_edges = get_edges(wave_points.value)
        
        fig.suptitle(f'Opacities for {RadtranGasDescriptor(self.gas_id, self.iso_id).label} at each point on the temperature grid.')
        
        for i in range(self.k.shape[2]):
            ax[i].pcolormesh(
                w_edges,
                p_edges,
                #self.k[:,:,i].T
                np.log(self.k[:,:,i].T)
            )
            
            if wave_points.unit == ans.enums.WaveUnit.Wavenumber_cm:
                ax[i].set_xlabel('Wavenumber ($cm^{-1}$)')
            elif wave_points.unit == ans.enums.WaveUnit.Wavelength_um:
                ax[i].set_xlabel(r'Wavelength ($\mu$m)')
            else:
                raise RuntimeError(f'Unknown {wave_unit}. Should be one of {ans.enums.WaveUnit.values()}')
            
            ax[i].set_xlabel('Wavenumber ($cm^{-1}$)')
            ax[i].set_ylabel('Pressure (atm)')
            ax[i].set_title(f'Temperature: {self.temp[i]} (K)')
        
        plt.show()



def read_legacy_header(f : str | io.IOBase) -> LblHeader:
    if not isinstance(f, io.IOBase):
        with open(f, 'rb') as g:
            return read_legacy_header(g)
    
    buf = f.read(_header_struct.size)    
    return LblHeader(*_header_struct.unpack_from(buf))


def read_legacy(
        f : str | io.ioBase, 
        wave_unit : None | ans.enums.WaveUnit =  None,
        fortran_record_byte_size = 4, # number of bytes in a fortran record, normally 4
) -> LblDataTProfilesAtPressure | LblDataTPGrid:
    """
    Read line-by-line table written in legacy format. NOTE: factor of 1E-20 (reverse of 
    factor applied when storing to avoid underflow in single precision floats) IS NOT 
    applied here, it is applied later in ForwardModel_0.
    
    ## ARGUMENTS ##
        
        f : str | ioBase
            Filename or FileLike object for table to read.
        
        wave_unit : None | ans.enums.WaveUnit
            Unit of wave data in file,. If None will infer from value of `LblHeader.vmin`; < 100 -> Wavelength_um, >= 100 -> Wavenumber_cm.
    
    ## RETURNS ##
    
        lbl_data : LblDataTProfilesAtPressure | LblDataTPGrid
            Line-by-line data in one of the two specified formats depending upon the 
            format specified in the file.
    """
    if not isinstance(f, io.IOBase):
        with open(f, 'rb') as g:
            return read_legacy(g, wave_unit)
    
    _lgr.debug(f'{f.tell()=}')
    hdr = read_legacy_header(f)
    _lgr.debug(f'{hdr=}')
    
    if wave_unit is None:
        if hdr.vmin < 100:
            wave_unit = ans.enums.WaveUnit.Wavelength_um
        else:
            wave_unit = ans.enums.WaveUnit.Wavenumber_um
        _lgr.warning(f'No `wave_unit` parameter given to {__name__}.read_legacy(...). Assuming "{wave_unit.name}" based off `LblHeader.vmin` = {hdr.vmin}')
    
    ptk = [
        None,
        None,
        None
    ]
    
    
    
    fortran_record_shape, p_struct, t_struct, k_struct = _get_body_structs(hdr.nwave, hdr.npress, hdr.ntemp)
    
    p_shape = (hdr.npress,)
    if hdr.ntemp < 0:
        t_shape = (hdr.npress, -hdr.ntemp)
        k_shape = (hdr.nwave, hdr.npress, -hdr.ntemp)
        lbl_data_type = LblDataTProfilesAtPressure
    else:
        t_shape = (hdr.ntemp,)
        k_shape = (hdr.nwave, hdr.npress, hdr.ntemp)
        lbl_data_type = LblDataTPGrid
    
    _lgr.debug(f'{p_shape=} {t_shape=} {k_shape=} {lbl_data_type=}')
    _lgr.debug(f'{f.tell()=}')
    
    # Read in pressure and temperature data, these start directly after the header
    for i, (x_struct, x_shape) in enumerate([(p_struct, p_shape), (t_struct, t_shape)]):
        _lgr.debug(f'{i=} {x_struct=} {x_shape=}')
        buf = f.read(x_struct.size)
        _lgr.debug(f'{f.tell()=}')
        
        ptk[i] = np.array(x_struct.unpack_from(buf), dtype=float, order='F').reshape(x_shape)
    
    # Absorption coefficients start at `hdr.irec0` records into the file. THIS IS NOT THE SAME AS AFTER THE PRESSURE AND TEMPERATURE DATA!!
    current_pos = f.tell()
    n_bytes_from_start_of_file_to_abs_coeff_data = fortran_record_byte_size * (hdr.irec0-1)
    
    if current_pos > n_bytes_from_start_of_file_to_abs_coeff_data:
        _lgr.warning(f'Header.IREC0 {hdr.irec0} is not correct for this file, attempting to guess the correct location for absorption coefficient data')
        f.seek(-k_struct.size, 2) # Assume the coefficients end and the end of the file and go backwards to where they should start
    else:
        _lgr.debug(f'Skipping to Header.IREC0 {hdr.irec0} bytes into the file as that is where the absorption coefficient data should be. {n_bytes_from_start_of_file_to_abs_coeff_data=}')
        f.seek(n_bytes_from_start_of_file_to_abs_coeff_data, 0)
    
    assert ptk[2] is None, 'Nothing should have been stored here before this point'
    ptk[2] = np.array(k_struct.unpack_from(f.read(k_struct.size)), dtype=float).reshape(k_shape)
    
    # Check we have read all the data correctly
    current_pos = f.tell()
    f.seek(0, 2)
    end_pos = f.tell()
    
    if current_pos != end_pos:
        _lgr.warning(f'We have not read all the data from the file, there are {end_pos - current_pos} bytes remaining. Using an alternate method to try and ensure we get the data correctly.')
        f.seek(-k_struct.size, 2) # Assume the coefficients end and the end of the file and go backwards to where they should start
        ptk[2] = np.array(k_struct.unpack_from(f.read(k_struct.size)), dtype=float).reshape(k_shape)
    
    return lbl_data_type(hdr.gas_id, hdr.iso_id, wave_unit, np.linspace(hdr.vmin, hdr.vmin+hdr.delv*hdr.nwave, hdr.nwave, endpoint=False), *ptk)
    
    








