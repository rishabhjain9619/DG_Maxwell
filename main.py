#! /usr/bin/env python3

import numpy as np
import arrayfire as af
import h5py
import scipy
from tqdm import trange

#af.set_backend('cpu')
#af.set_device(0)

from dg_maxwell.tests import test_waveEqn
from matplotlib import pyplot as pl
from dg_maxwell import params
from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell.tests import convergence_tests 
from dg_maxwell import utils


pl.rcParams['figure.figsize']     = 12, 7.5
pl.rcParams['lines.linewidth']    = 1.5
pl.rcParams['font.family']        = 'serif'
pl.rcParams['font.weight']        = 'bold'
pl.rcParams['font.size']          = 20  
pl.rcParams['font.sans-serif']    = 'serif'
pl.rcParams['text.usetex']        = True
pl.rcParams['axes.linewidth']     = 1.5
pl.rcParams['axes.titlesize']     = 'medium'
pl.rcParams['axes.labelsize']     = 'medium'

pl.rcParams['xtick.major.size']   = 8
pl.rcParams['xtick.minor.size']   = 4
pl.rcParams['xtick.major.pad']    = 8
pl.rcParams['xtick.minor.pad']    = 8
pl.rcParams['xtick.color']        = 'k'
pl.rcParams['xtick.labelsize']    = 'medium'
pl.rcParams['xtick.direction']    = 'in'    

pl.rcParams['ytick.major.size']   = 8
pl.rcParams['ytick.minor.size']   = 4
pl.rcParams['ytick.major.pad']    = 8
pl.rcParams['ytick.minor.pad']    = 8
pl.rcParams['ytick.color']        = 'k'
pl.rcParams['ytick.labelsize']    = 'medium'
pl.rcParams['ytick.direction']    = 'in'
pl.rcParams['text.usetex']        = True
pl.rcParams['text.latex.unicode'] = True

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
 
    #error_arr_E_z = []
    #error_arr_B_y = []
    #for n_lgl in range(4, 30):
        #test_waveEqn.change_parameters(n_lgl, params.N_Elements, n_lgl)
        #E_z     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
        #B_y     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
        #u_init  =  af.join(2, E_z, B_y)
        #u_fin   =  wave_equation.mod_time_evolution(u_init)
        #u_diff  =  wave_equation.E_z_B_y_diff(u_fin, params.time)
        #u_diff_E_z = u_diff[:,:,0]
        #u_diff_B_y = u_diff[:,:,1]
        #error_E_z  = scipy.integrate.trapz(np.ravel(af.transpose(u_diff_E_z)), np.ravel(af.transpose(params.element_LGL)))
        #error_B_y  = scipy.integrate.trapz(np.ravel(af.transpose(u_diff_B_y)), np.ravel(af.transpose(params.element_LGL)))
        #error_arr_E_z.append(error_E_z)
        #error_arr_B_y.append(error_B_y)
    #error_arr_B_y = np.asarray(error_arr_B_y)
    #error_arr_E_z = np.asarray(error_arr_E_z)
    #np.savetxt('results/L1_norm_error_arr_B_y.csv', error_arr_B_y, delimiter = ',')
    #np.savetxt('results/L1_norm_error_arr_E_z.csv', error_arr_E_z, delimiter = ',')
    #pl.semilogy(np.arange(17,20), np.asarray(error_arr_E_z), '.-',)
    #pl.title(r'Plot for L_1 norm')
    #pl.xlabel(r'No. of lgl points')
    #pl.ylabel(r'Error norm')
    #pl.legend(prop={'size': 14})
    #pl.show()
    
    content = [1.616192092248700885e-03,
                9.814033651037902951e-05,
                4.860018303330410675e-06,
                2.197414430464293444e-07,
                5.127674505798240752e-08,
                1.433756586547475145e-08,
                7.123674858711929454e-09,
                9.786046114072318074e-09,
                2.061857130826404538e-08,
                2.109129626631023750e-08,
                6.372128495786982544e-09,
                2.025596488447508045e-08,
                1.969737657388321995e-08,
                2.344363252870452880e-08,
                3.375051906897725237e-08,
                1.483019392675207886e-07,
                7.305573875438503111e-08,
                1.046600292294862190e-06,
                3.157475475772883602e-06,
                8.610247749159001264e-06,
                9.737043452411323202e-06,
                1.575376727169526113e-04,
                3.131124337603548844e-04,
                4.291061776793542561e-04,
                2.264000672147915404e-03,
                1.859007376449558352e-02
                ]
    
    content = np.asarray(content)
    pl.semilogy(np.arange(4,30), content, '.-',)
    pl.title(r'Plot for L_1 norm')
    pl.xlabel(r'No. of lgl points')
    pl.ylabel(r'Error norm')
    pl.legend(prop={'size': 14})
    pl.show()
    
    
    
    
    #content = utils.plotcsv()
    #content_2 = np.array(content[:12], dtype = np.float64).flatten()
    #pl.loglog(np.arange(4, 16), content_2, '.-', label = r'Convergence plot')
    
    #N = np.arange(4, 9, dtype = np.float64)
    #pl.loglog(N, (N * 0.5)**(-1.5 * N), '--', label = r'$N^{-N}$')    
    #pl.title(r'Convergence plot for $B_y$ $(N_{quad} = N_{LGL})$')
    #pl.xlabel(r'N_{LGL}')
    #pl.ylabel(r'L1 Norm')
    #pl.legend(prop={'size': 14})
    
    #pl.show()
    #n_lgl = 10
    #test_waveEqn.change_parameters(n_lgl, params.N_Elements, n_lgl)
    #E_z     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
    #B_y     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
    #u_init  =  af.join(2, E_z, B_y)
    #u_fin   =  wave_equation.mod_time_evolution(u_init)
    #E_z_fin = u_fin[:,:,0]
    #B_y_fin = u_fin[:,:,1]
    #time_2  = params.time[-1].to_array()
    #E_z_t    =  (np.cos(2*np.pi*time_2[0])-np.sin(2*np.pi*time_2[0]))*af.sin(2 * np.pi * params.element_LGL) + (np.cos(2*np.pi*time_2[0])+np.sin(2*np.pi*time_2[0]))*af.cos(2 * np.pi * params.element_LGL)
    #B_y_t    =  (np.cos(2*np.pi*time_2[0])-np.sin(2*np.pi*time_2[0]))*af.sin(2 * np.pi * params.element_LGL) + (np.cos(2*np.pi*time_2[0])+np.sin(2*np.pi*time_2[0]))*af.cos(2 * np.pi * params.element_LGL)
    #error   = af.abs(E_z_fin- E_z_t)
    #pl.semilogy(params.element_LGL, error, '.-',)
    #n_lgl = 24
    #test_waveEqn.change_parameters(n_lgl, params.N_Elements, n_lgl)
    #E_z     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
    #B_y     =  af.sin(2 * np.pi * params.element_LGL) + af.cos(2 * np.pi * params.element_LGL)
    #u_init  =  af.join(2, E_z, B_y)
    #u_fin   =  wave_equation.mod_time_evolution(u_init)
    #E_z_fin = u_fin[:,:,0]
    #B_y_fin = u_fin[:,:,1]
    #time_2  = params.time[-1].to_array()
    #E_z_t    =  (np.cos(2*np.pi*time_2[0])-np.sin(2*np.pi*time_2[0]))*af.sin(2 * np.pi * params.element_LGL) + (np.cos(2*np.pi*time_2[0])+np.sin(2*np.pi*time_2[0]))*af.cos(2 * np.pi * params.element_LGL)
    #B_y_t    =  (np.cos(2*np.pi*time_2[0])-np.sin(2*np.pi*time_2[0]))*af.sin(2 * np.pi * params.element_LGL) + (np.cos(2*np.pi*time_2[0])+np.sin(2*np.pi*time_2[0]))*af.cos(2 * np.pi * params.element_LGL)
    #error_2   = af.abs(E_z_fin- E_z_t)
    #pl.plot(params.element_LGL, E_z_t, '--',)
    #pl.title(r'Plot for E_z')
    #pl.xlabel(r'Elements')
    #pl.ylabel(r'Amplitude of E_z')
    ##pl.legend(prop={'size': 14})
    #pl.show()