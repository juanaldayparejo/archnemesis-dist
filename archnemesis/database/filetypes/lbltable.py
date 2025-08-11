from __future__ import annotations #  for 3.9 compatability

import io
from typing import NamedTuple, TYPE_CHECKING, Self, ClassVar
import struct


import numpy as np
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    NPRESS = "Number of points in pressure grid"
    NTEMP_PER_PRESSURE = "Number of temperatures per pressure point"
    NTEMP = "Number of points in temperature grid"


_header_struct = struct.Struct('<2i2f4i')

def _get_body_structs(nwave, npress, ntemp):
    print(f'{nwave=} {npress=} {ntemp=} {nwave * npress * ntemp=}')
    return (
        struct.Struct(f'<{npress}f'),
        struct.Struct(f'<{ntemp if ntemp > 0 else -ntemp*npress}f'),
        struct.Struct(f'<{nwave * npress * abs(ntemp)}f'),
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


class LblDataFormat1(NamedTuple):
    gas_id : int
    iso_id : int
    wave : np.ndarray[['NWAVE'], float]
    press : np.ndarray[['NPRESS'], float]
    temp : np.ndarray[['NPRESS', 'NTEMP_PER_PRESSURE'], float]
    k : np.ndarray[['NWAVE','NPRESS','NTEMP_PER_PRESSURE'], float]
    
    def write_legacy_header(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy_header(g)
        
        f.write(_header_struct.pack(
            9, # always 9 for some reason
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
                return self.write_legacy_header(g)
        
        self.write_legacy_header(f)
        
        p_struct, t_struct, k_struct = _get_body_structs(self.wave.size, self.press.size, self.temp.shape[1])
        
        f.write(
            p_struct.pack(
                self.press
            )
        )
        
        f.write(
            t_struct.pack(
                self.temp
            )
        )
        
        f.write(
            l_struct.pack(
                self.k
            )
        )
    
    def plot(self):
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
            
        p_edges = np.exp(get_edges(np.log(self.press)))
        w_edges = get_edges(self.wave)
        for i in range(self.k.shape[2]):
            ax[i].pcolormesh(
                w_edges,
                p_edges,
                #self.k[:,:,i].T
                np.log(self.k[:,:,i].T)
            )
            ax[i].set_xlabel('Wavenumber ($cm^{-1}$)')
            ax[i].set_ylabel('Pressure (atm)')
            ax[i].set_yscale('log')
            ax[i].invert_yaxis()
            
            ax2[i].plot(self.temp[:,i], self.press, color='red', linestyle='--', alpha=0.6)
            ax2[i].set_xlabel('Temperature (K)')
        
        plt.show()
            
        
        

class LblDataFormat2(NamedTuple):
    gas_id : int
    iso_id : int
    wave : np.ndarray[['NWAVE'], float]
    press : np.ndarray[['NPRESS'], float]
    temp : np.ndarray[['NTEMP'], float]
    k : np.ndarray[['NWAVE','NPRESS','NTEMP'], float]


    def write_legacy_header(self, f : str | io.IOBase):
        if not isinstance(f, io.IOBase):
            with open(f, 'wb') as g:
                return self.write_legacy_header(g)
        
        f.write(_header_struct.pack(
            9, # always 9 for some reason
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
                return self.write_legacy_header(g)
        
        self.write_legacy_header(f)
        
        p_struct, t_struct, k_struct = _get_body_structs(self.wave.size, self.press.size, self.temp.size)
        
        f.write(
            p_struct.pack(
                self.press
            )
        )
        
        f.write(
            t_struct.pack(
                self.temp
            )
        )
        
        f.write(
            l_struct.pack(
                self.k
            )
        )

    def plot(self):
        fig, ax = plt.subplots(self.k.shape[2],1, figsize=(12,6*self.k.shape[2]), squeeze=False)
        ax = ax.flatten()
        
        for i in range(self.k.shape[2]):
            ax[i].imshow(np.log(self.k[:,:,i].T), aspect='auto')
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


def read_legacy(f : str | ioBase) -> LblDataFormat1 | LblDataFormat2:
    if not isinstance(f, io.IOBase):
        with open(f, 'rb') as g:
            return read_legacy(g)
    
    hdr = read_legacy_header(f)
    
    ptk = [
        None,
        None,
        None
    ]
    
    p_struct, t_struct, k_struct = _get_body_structs(hdr.nwave, hdr.npress, hdr.ntemp)
    
    if hdr.ntemp < 0:
        
        for i, (x_struct, x_shape) in enumerate([(p_struct, (hdr.npress,)), (t_struct, (hdr.npress, -hdr.ntemp,)), (k_struct, (hdr.nwave, hdr.npress, -hdr.ntemp))]):
            print(f'{i=} {x_struct=} {x_shape=}')
            buf = f.read(x_struct.size)
            ptk[i] = np.array(x_struct.unpack_from(buf), dtype=float).reshape(x_shape)
        
        return LblDataFormat1(hdr.gas_id, hdr.iso_id, np.linspace(hdr.vmin, hdr.vmin+hdr.delv*hdr.nwave, hdr.nwave, endpoint=False), *ptk)

    else:
        
        for i, (x_struct, x_shape) in enumerate([(p_struct, (hdr.npress,)), (t_struct, (hdr.ntemp,)), (k_struct, (hdr.nwave, hdr.npress, hdr.ntemp))]):
            buf = f.read(x_struct.size)
            ptk[i] = np.array(x_struct.unpack_from(buf), dtype=float).reshape(x_shape)
        
        return LblDataFormat2(hdr.gas_id, hdr.iso_id, np.linspace(hdr.vmin, hdr.vmin+hdr.delv*hdr.nwave, hdr.nwave, endpoint=False), *ptk)
    
    

    