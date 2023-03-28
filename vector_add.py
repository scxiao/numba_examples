from numba import roc
import numpy as np

@roc.jit(device=True)
def a_device_function(a, b):
    return a + b

@roc.jit
def kernel(an_array):
    pos = roc.get_global_id(0)
    if (pos < an_array.size):
        an_array[pos] = a_device_function(1, pos)

n = 64
x = np.zeros(n)
kernel[1, n](x)

print(f'x = {x}')

