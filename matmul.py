import numpy as np
from numba import roc
from numba import float32
from time import time as timer

blocksize = 16
gridsize = 16

@roc.jit('(float32[:,:], float32[:,:], float32[:,:])')
def matmulfast(A, B, C):
    x = roc.get_global_id(0)
    y = roc.get_global_id(1)

    tx = roc.get_local_id(0)
    ty = roc.get_local_id(1)

    sA = roc.shared.array(shape=(blocksize, blocksize), dtype=float32)
    sB = roc.shared.array(shape=(blocksize, blocksize), dtype=float32)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0

    for i in range(gridsize):
        # preload
        sA[tx, ty] = A[x, ty + i * blocksize]
        sB[tx, ty] = B[tx + i * blocksize, y]
        # wait for preload to end
        roc.barrier(1)
        # compute loop
        for j in range(blocksize):
            tmp += sA[tx, j] * sB[j, ty]
        # wait for compute to end
        roc.barrier(1)

    C[x, y] = tmp

N = gridsize * blocksize
A = np.random.random((N, N)).astype(np.float32)
B = np.random.random((N, N)).astype(np.float32)
C = np.zeros_like(A)

griddim = gridsize, gridsize
blockdim = blocksize, blocksize

with roc.register(A, B, C):
    ts = timer()
    matmulfast[griddim, blockdim](A, B, C)
    te = timer()
    print("1st GPU time:", te - ts)

with roc.register(A, B, C):
    ts = timer()
    matmulfast[griddim, blockdim](A, B, C)
    te = timer()
    print("2nd GPU time:", te - ts)

ts = timer()
ans = np.dot(A, B)
te = timer()
print("CPU time:", te - ts)
np.testing.assert_allclose(ans, C, rtol=1e-5)

