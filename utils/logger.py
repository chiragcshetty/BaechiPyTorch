from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

CRITICAL=logging.CRITICAL
ERROR=logging.ERROR
WARNING=logging.WARNING
INFO=logging.INFO
DEBUG=logging.DEBUG

#open('runtime.log', 'w').close() # clear the file
logging.basicConfig(filename='runtime.log', level=logging.INFO )
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def get_logger(filepath, level=INFO):
    name = os.path.splitext(os.path.basename(filepath))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger
