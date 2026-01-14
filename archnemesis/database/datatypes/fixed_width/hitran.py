

from .base import AsciiFixedWidthFormat


class FormatHitran160(AsciiFixedWidthFormat):
    """
    Fixed width format can be specified like a table with the following columns:
    
    attribute name                     : type                           = width (in ascii characters)
    """
    gas_id                             : int                            = 2
    iso_id                             : int                            = 1
    line_wavenumber                    : float                          = 12
    line_strength                      : float                          = 10
    einstein_a_coeff                   : float                          = 10
    gamma_amb                          : float                          = 5
    gamma_self                         : float                          = 5
    e_lower                            : float                          = 10 
    n_amb                              : float                          = 4
    delta_amb                          : float                          = 8
    global_upper_quanta                : str                            = 15
    global_lower_quanta                : str                            = 15
    local_upper_quanta                 : str                            = 15
    local_lower_quanta                 : str                            = 15
    ierr                               : tuple[int,int,int,int,int,int] = (1,1,1,1,1,1)
    iref                               : tuple[int,int,int,int,int,int] = (2,2,2,2,2,2)
    line_mixing_flag                   : str                            = 1
    gp                                 : float                          = 7
    gpp                                : float                          = 7


class FormatHitran100(AsciiFixedWidthFormat):
    """
    Fixed width format can be specified like a table with the following columns:
    
    attribute name                     : type               = width (in ascii characters)
    """
    gas_id                             : int                = 2
    iso_id                             : int                = 1
    line_wavenumber                    : float              = 12
    line_strength                      : float              = 10
    einstein_a_coeff                   : float              = 10
    gamma_amb                          : float              = 5
    gamma_self                         : float              = 5
    e_lower                            : float              = 10 
    n_amb                              : float              = 4
    delta_amb                          : float              = 8
    global_upper_quanta                : str                = 3
    global_lower_quanta                : str                = 3
    local_upper_quanta                 : str                = 9
    local_lower_quanta                 : str                = 9
    ierr                               : tuple[int,int,int] = (1,1,1)
    iref                               : tuple[int,int,int] = (2,2,2)




class FormatLegacyHitran160(AsciiFixedWidthFormat):
    """
    Fixed width format can be specified like a table with the following columns:
    
    attribute name                     : type               = width (in ascii characters)
    """
    gas_id                             : int                            = 2
    iso_id                             : int                            = 1
    line_wavenumber                    : float                          = 12
    line_strength                      : float                          = 10
    weighted_transition_moment_squared : float                          = 10
    gamma_amb                          : float                          = 5
    gamma_self                         : float                          = 5
    e_lower                            : float                          = 10 
    n_amb                              : float                          = 4
    delta_amb                          : float                          = 8
    global_upper_quanta                : str                            = 15
    global_lower_quanta                : str                            = 15
    local_upper_quanta                 : str                            = 15
    local_lower_quanta                 : str                            = 15
    ierr                               : tuple[int,int,int,int,int,int] = (1,1,1,1,1,1)
    iref                               : tuple[int,int,int,int,int,int] = (2,2,2,2,2,2)
    line_mixing_flag                   : str                            = 1
    gp                                 : float                          = 7
    gpp                                : float                          = 7



class FormatLegacyHitran100(AsciiFixedWidthFormat):
    """
    Fixed width format can be specified like a table with the following columns:
    
    attribute name                     : type               = width (in ascii characters)
    """
    gas_id                             : int                = 2
    iso_id                             : int                = 1
    line_wavenumber                    : float              = 12
    line_strength                      : float              = 10
    weighted_transition_moment_squared : float              = 10
    gamma_amb                          : float              = 5
    gamma_self                         : float              = 5
    e_lower                            : float              = 10 
    n_amb                              : float              = 4
    delta_amb                          : float              = 8
    global_upper_quanta                : str                = 3
    global_lower_quanta                : str                = 3
    local_upper_quanta                 : str                = 9
    local_lower_quanta                 : str                = 9
    ierr                               : tuple[int,int,int] = (1,1,1)
    iref                               : tuple[int,int,int] = (2,2,2)




