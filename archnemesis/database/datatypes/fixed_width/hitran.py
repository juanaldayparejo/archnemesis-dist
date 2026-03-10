

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

    @classmethod
    def get_record_from_str(cls, s : str):
        # Custom reader is always faster than relying on the generic one.
        return cls._record_type(
            int(s[:2]),
            int(s[2:3]),
            float(s[3:15]),
            float(s[15:25]),
            float(s[25:35]),
            float(s[35:40]),
            float(s[40:45]),
            float(s[45:55]),
            float(s[55:59]),
            float(s[59:67]),
            s[67:82],
            s[82:97],
            s[97:112],
            s[112:127],
            tuple(map(int, s[127:133])),
            tuple(map(int, (s[133:135],s[135:137],s[137:139],s[139:141],s[141:143],s[143:145]))),
            s[145:146],
            float(s[146:153]),
            float(s[153:160])
        )

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




