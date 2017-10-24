"""
Machine learning configuration file.
"""
from sklearn.preprocessing import *

def size_classes(size):
    """
    Definitions of size classes:
    2048   >= size => 0      -> 'small'
    4096   >= size  > 2048   -> 'medium'
    16384  >= size  > 4096   -> 'large'
    131072 >= size  > 16384  -> 'huge'
    infty  >= size  > 131072 -> 'enormous'
    """
    if (size >= 0 and size <= 2048):
        return 'small'
    elif (size > 2048 and size <= 4096):
        return 'medium'
    elif (size > 4096 and  size <= 32768):
        return 'large'
    elif (size > 32768 and size <= 131072):
        return 'huge'
    else:
        return 'enormous'

# The number of input lines to use for the
num_input_lines = 10

# The scaler to use for normalization.
scaler = QuantileTransformer(output_distribution='normal')
