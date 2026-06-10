# Preliminary

# Set environment variables - not mandatory but recommended if you use Parallel()
#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'


import wannierberri as wberri
print (f"Using WannierBerri version {wberri.__version__}")
import numpy as np
import scipy
import matplotlib.pyplot as plt
from termcolor import cprint

#  This block is needed if you run this cell for a second time
#  because one cannot initiate two parallel environments at a time
# try:
#     parallel.shutdown()
# except NameError:
#     pass

# Chiose one of the options:

# parallel = wberri.Parallel(num_cpus=2)
#parallel = wberri.Parallel()  # automatic detection
#parallel = wberri.Serial()

system=wberri.System_tb('la3ir3o11_tb.dat')
