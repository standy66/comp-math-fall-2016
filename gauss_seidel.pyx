from typing import Union
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gauss_seidel_solve_heat_eq(np.ndarray[np.float32_t, ndim=3] phi, float lambda_x, float lambda_y,
                               float dx, float dy, float dt, py_n_iters : Union[str, int] = "adaptive",
                               double rmse_epsilon = 1e-7):
    cdef int Nt = phi.shape[0], Nx = phi.shape[1], Ny = phi.shape[2]
    cdef float TX = 1.0 / dx * lambda_x / dx, TY = 1.0 / dy * lambda_y / dy
    cdef float c = -TX
    cdef float b = -TX
    cdef float g = -TY
    cdef float f = -TY
    cdef float gamma = 1.0 / dt
    cdef float a = - (c + b + g + f) + gamma
    cdef int n_iters
    if py_n_iters == "adaptive":
        n_iters = -1
    else:
        n_iters = py_n_iters
    cdef int t, k, i, j
    cdef float mse = 0.0
    cdef float new_val
    for t in range(1, Nt):
        k = 0
        while True:
            mse = 0.0
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    new_val = 1.0 / a * (gamma * phi[t - 1, i, j] - c * phi[t, i - 1, j] - g * phi[t, i, j - 1]
                                              - b * phi[t, i + 1, j] - f * phi[t, i, j + 1])
                    mse += (new_val - phi[t, i, j]) ** 2
                    phi[t, i, j] = new_val
            k += 1
            if n_iters == -1:
                rmse = sqrt(mse / ((Nx - 2) * (Ny - 2)))
                if rmse < rmse_epsilon:
                    print("Finished timestep {} after {} iterations, adaptive rmse {}".format(t, k, rmse))
                    break
            elif k == n_iters:
                print("Finished timestep {} after {} iterations".format(t, k))
                break
