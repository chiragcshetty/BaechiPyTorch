import sys, os
## To let modules within this folder import each other without having to know the callers path 
## i.e to use "import util_functions", instead of 
## "import {caller_path}.baechi.core.utils_fucntion"
sys.path.append(os.path.dirname(__file__))

## https://stackoverflow.com/questions/9066777/howto-import-modules-with-dependencies-in-the-same-absolute-relative-path-with-i