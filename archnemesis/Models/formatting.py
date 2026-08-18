import numpy as np

def format_float(x : float, width : int = 8):
    nlog = int(np.fix(np.log10(np.abs(x))))
    if nlog >= 100 or nlog <= -100:
        exp_digits = 3
    else:
        exp_digits = 2
    exp_width = exp_digits+2
    
    if nlog > width - (exp_width+1):
        return f'{{:> {width}.{width-3-exp_width}E}}'.format(x)
    if nlog < -1*exp_width:
        return f'{{:> {width}.{width-3-exp_width}E}}'.format(x)
    if nlog <= 0:
        return f'{{:> {width}.{width-3}f}}'.format(x)
    else:
        return f'{{:> {width}.{width-3-nlog}f}}'.format(x)
