#! /usr/bin/env python3

import numpy as np
import arrayfire as af
import h5py
from tqdm import trange

#af.set_backend('cpu')
#af.set_device(0)

from dg_maxwell.tests import test_waveEqn
from matplotlib import pyplot as pl
from dg_maxwell import params
from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell.tests import convergence_tests 

if __name__ == '__main__':
    # 1. Set the initial conditions
    
    ## 1. Sin
    #E_00 = 1.
    #E_01 = 1.

    #B_00 = 0.2
    #B_01 = 0.5

    #E_z_init = E_00 * af.sin(2 * np.pi * params.element_LGL) \
            #+ E_01 * af.cos(2 * np.pi * params.element_LGL)

    #B_y_init = B_00 * af.sin(2 * np.pi * params.element_LGL) \
            #+ B_01 * af.cos(2 * np.pi * params.element_LGL)
    
    ### 2. Gaussian
    #E_0 = 1.
    #B_0 = 1.
    
    #sigma = 0.1
    
    #E_z_init = E_0 * np.e**(-(params.element_LGL)**2 / sigma**2)
    #B_y_init = B_0 * np.e**(-(params.element_LGL)**2 / sigma**2)
    
    
    ################################################################
    ######################## SET u_init ############################
    ################################################################
    #u_init = af.constant(0., d0 = params.N_LGL, d1 = params.N_Elements, d2 = 2)
    #u_init[:, :, 0] = E_z_init
    #u_init[:, :, 1] = B_y_init
    
    #u_init = params.u_init
 
    error_arr_E_z = []
    error_arr_B_y = []
    for n_lgl in range(4,25):
        test_waveEqn.change_parameters(n_lgl, params.N_Elements, n_lgl)
        E_z     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
        B_y     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
        u_init  =  af.join(2, E_z, B_y)
        u_fin   =  wave_equation.mod_time_evolution(u_init)
        u_diff  =  wave_equation.E_z_B_y_diff(u_fin, params.time)
        u_diff_E_z = u_diff[:,:,0]
        u_diff_B_y = u_diff[:,:,1]
        error_E_z  = convergence_tests.L1_norm(u_diff_E_z)
        error_B_y  = convergence_tests.L1_norm(u_diff_B_y)
        error_arr_E_z.append(error_E_z)
        error_arr_B_y.append(error_B_y)
    print(error_arr_B_y)
    print(error_arr_E_z)
    
    np.savetxt('results/L1_norm_error_arr_B_y.csv', error_arr_B_y, delimiter = ',')
    np.savetxt('results/L1_norm_error_arr_E_z.csv', error_arr_E_z, delimiter = ',')
    pl.loglog(np.array(4, 25), error_arr_E_z, '-')
    pl.loglog(np.array(4, 25), error_arr_B_y, '--')
    
    pl.show()