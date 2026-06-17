# Borrowed from https://github.com/scikit-hep/numba-stats/blob/main/src/numba_stats/_special.py
#
# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from typing import Any

from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float64


def get(name: str, signature: Any) -> Any:
    # create new function object with correct signature that numba can call
    from scipy.special import cython_special

    # scipy-1.12 started to provide fused versions for some special functions
    if name in {"betainc", "stdtr", "stdtrit"}:
        fuse_name = f"__pyx_fuse_0{name}"
    else:
        fuse_name = f"__pyx_fuse_1{name}"
    if fuse_name not in cython_special.__pyx_capi__:
        fuse_name = name

    addr = get_cython_function_address("scipy.special.cython_special", fuse_name)

    # dynamically create type that inherits from WrapperAddressProtocol
    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()

################################ LINE PROFILE FUNCTIONS ###################################

# VOIGT PROFILE ##########################################################
voigt_profile = get("voigt_profile", float64(float64, float64, float64)) #
# ARGUMENTS -------------------------------------------------------------#
#   x - Value to calculate probability for                               #
#   sigma - Standard deviation of normal distribution part               #
#   gamma - Half-width at Half-maximum of Cauchy distribution part       #
##########################################################################