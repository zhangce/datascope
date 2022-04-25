#!python
#cython: language_level=3

import numpy as np


cimport numpy as np
cimport cython
np.import_array()

DTYPE = np.float
ctypedef np.float_t DTYPE_t


# def say_hello_to(name):
#     print("Hello %s!" % name)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.profile(True)
def compute_all_importances_cy(np.ndarray[DTYPE_t, ndim=2] unit_distances, np.ndarray[DTYPE_t, ndim=2] unit_utilities, np.ndarray[DTYPE_t, ndim=1] null_scores):

    assert unit_distances.dtype == DTYPE and unit_utilities.dtype == DTYPE

    # Compute unit importances.
    cdef int n_units = unit_distances.shape[0]
    cdef int n_units_p = n_units + 1
    cdef int n_test = unit_distances.shape[1]
    cdef np.ndarray[np.int_t, ndim=2] idxs
    cdef np.ndarray[DTYPE_t, ndim=1] all_importances
    cdef int i, j
    cdef int i_1, i_2
    cdef float current

    all_importances = np.zeros([n_units + 1], dtype=DTYPE)
    unit_utilities = np.vstack((unit_utilities, null_scores))
    idxs = np.vstack((np.argsort(unit_distances, axis=0), np.full((1, n_test), n_units, dtype=int)))

    for j in range(n_test):
        current = 0.0
        for i in range(n_units - 1, -1, -1):
            i_1 = idxs[i, j]
            i_2 = idxs[i + 1, j]
            current += (unit_utilities[i_1, j] - unit_utilities[i_2, j]) / (i + 1)
            all_importances[i_1] += current
    return all_importances[:-1] / n_test
