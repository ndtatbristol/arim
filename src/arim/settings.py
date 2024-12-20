"""
Settings use in this library.

Usage:

    import settings as s

    # Get parameter:
    print(s.SOME_PARAMETER)

    # Change parameter:
    s.SOME_PARAMETER = 'new_value'

    # Alt access (the parameter is a string)
    print(s.__dict__['SOME_PARAMETER'])


"""

from multiprocessing import cpu_count

import numpy as np

# ------------------------------------------------------------------------------
## Standard types
# Unreliable, because inconsistently used in this library. In the future, using
# numpy's default, i.e. np.dtype(float), np.dtype(int), etcseems more realistic.
FLOAT = np.dtype(float)
INT = np.int32
UINT = np.uint32
COMPLEX = np.dtype(complex)
DATETIME = "%Y-%m-%d %H:%M:%S"  # call: datetime.datetime.now().strftime(DATETIME)

# ------------------------------------------------------------------------------
## Default for computation

# In the context of a multithreaded computation, BLOCK_SIZE is the typical number of
# floats/integers assigned to each thread. It depends on the function itself and
# the CPU. Adjusting these values can reduce the computational time.
# These values are OK for a Intel Core i7-4790 (L1: 64 KB per core, L2: 256 KB per core, L3: 8 MB shared).
BLOCK_SIZE_EUC_DISTANCE = 500
BLOCK_SIZE_DELAY_AND_SUM = 500
BLOCK_SIZE_FIND_MIN_TIMES = 50000
NUMTHREADS = cpu_count()


# ------------------------------------------------------------------------------
## Types for HDF5
H5_FLOAT = FLOAT
H5_INT = INT
H5_COMPLEX = COMPLEX
H5_DATETIME = DATETIME
