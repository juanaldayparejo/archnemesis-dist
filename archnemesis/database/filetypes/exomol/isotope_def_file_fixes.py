"""
Some EXOMOL isotope definition files have errors that need to be fixed.
"""
from typing import Callable
import re

def fix_h2o_wat_uv296_def_file(file_contents : str) -> str:
	file_contents = re.sub(
		r"6v(\s*)# No\. of quanta defined",
		r"6 \1# No. of quanta defined",
		file_contents
	)
	return file_contents


# Mappint form EXOMOL isotope definition file URL to a function that takes
# the file contents as a string and returns the fixed file contents as a string.
iso_def_file_fix_map : dict[str, Callable[[str], str]] = {
	"https://www.exomol.com/db/H2O/1H2-16O/WAT-UV296/1H2-16O__WAT-UV296.def" : fix_h2o_wat_uv296_def_file,
	"https://www.exomol.com/db/H2O/1H2-16O/CKYKKY/1H2-16O__CKYKKY.def" : fix_h2o_wat_uv296_def_file, # same fix as above

}