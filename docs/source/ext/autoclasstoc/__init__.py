# This contains code from package `autoclasstoc` (https://github.com/kalekundert/autoclasstoc)
# The original package follows the MIT license, see LICENSE.txt in the same folder
# It has been modified for the package `flory` by Yicheng Qiang.
# The `flory` package also follows the MIT license, see LICENSE in the root directory of the package.

"""\
Add a succinct TOC to auto-documented classes.
"""

__version__ = "1.6.0"

# Make this package appear flat to external tools (e.g. sphinx):
from inspect import isclass, isfunction

from . import utils
from .errors import *
from .plugin import *
from .sections import *

for obj in locals().copy().values():
    if isfunction(obj) or isclass(obj):
        obj.__module__ = "autoclasstoc"

del isfunction, isclass, obj
