#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
import h5py
from tqdm import trange
from matplotlib import pyplot as pl
from tqdm import trange

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import utils

def A_matrix(advec_var):
    '''
    '''
    jacobian = 100.
    A_ij = wave_equation_2d.A_matrix(params.N_LGL, advec_var) / jacobian

    return A_ij

def volume_integral(u, advec_var):
    '''
    Vectorize, p, q, moddims.
    '''
    dLp_xi_ij_Lq_eta_ij = advec_var.dLp_Lq
    dLq_eta_ij_Lp_xi_ij = advec_var.dLq_Lp
    dxi_dx   = 10.
    deta_dy  = 10.
    jacobian = 100.
    c_x = params.c_x
    c_y = params.c_y

    if (params.volume_integrand_scheme_2d == 'Lobatto' and params.N_LGL == params.N_quad):
        w_i = af.flat(af.transpose(af.tile(advec_var.lobatto_weights_quadrature, 1, params.N_LGL)))
        w_j = af.tile(advec_var.lobatto_weights_quadrature, params.N_LGL)
        wi_wj = w_i * w_j
        wi_wj_dLp_xi = af.broadcast(utils.multiply, wi_wj, advec_var.dLp_Lq)
        
        volume_integrand_ij_1_sp = c_x * dxi_dx * af.broadcast(utils.multiply,
                                                               wi_wj_dLp_xi,
                                                               u) \
                                 / jacobian
        wi_wj_dLq_eta = af.broadcast(utils.multiply, w_i * w_j, advec_var.dLq_Lp)
        volume_integrand_ij_2_sp = c_y * deta_dy * af.broadcast(utils.multiply,\
                                               wi_wj_dLq_eta, u) / jacobian

        volume_integral = af.reorder(af.sum(volume_integrand_ij_1_sp + volume_integrand_ij_2_sp, 0), 2, 1, 0)

    else:
        volume_integrand_ij_1 = c_x * dxi_dx * af.broadcast(utils.multiply,\
                                        dLp_xi_ij_Lq_eta_ij,\
                                        u) / jacobian

        volume_integrand_ij_2 = c_y * deta_dy * af.broadcast(utils.multiply,\
                                        dLq_eta_ij_Lp_xi_ij,\
                                        u) / jacobian

        volume_integrand_ij = af.moddims(volume_integrand_ij_1 + volume_integrand_ij_2, params.N_LGL ** 2,\
                                         (params.N_LGL ** 2) * 100)

        lagrange_interpolation = af.moddims(wave_equation_2d.lag_interpolation_2d(volume_integrand_ij, advec_var.Li_Lj_coeffs),
                                            params.N_LGL, params.N_LGL, params.N_LGL ** 2  * 100)

        volume_integrand_total = utils.integrate_2d_multivar_poly(lagrange_interpolation[:, :, :],\
                                                    params.N_quad,'gauss', advec_var)
        volume_integral        = af.transpose(af.moddims(volume_integrand_total, 100, params.N_LGL ** 2))

    return volume_integral


def lax_friedrichs_flux(u_n, u_n_plus_1):
    '''
    Calculates the Lax-Friedrichs flux.
    [TODO] Documentation needed
    '''
    lf_flux = None
    
    lf_flux = (wave_equation_2d.F_x(u_n_plus_1) \
            + wave_equation_2d.F_x(u_n)) / 2. \
            - params.c_lax * (u_n_plus_1 - u_n) / 2.
    
    return lf_flux

def u_at_edge(u_e_ij, edge_id):
    '''
    Finds the :math:`u` at given edge id for :math:`u_{eij}`.
    
    [Tested]
    [TODO] Documentation needed
    '''
    u_e_ij = af.moddims(u_e_ij, params.N_LGL, params.N_LGL)
    
    if edge_id == 0:
        return u_e_ij[:, 0]
    
    elif edge_id == 1:
        return u_e_ij[-1]
    
    elif edge_id == 2:
        return u_e_ij[:, -1]
    
    elif edge_id == 3:
        return u_e_ij[0]
    
    else:
        return
    
    return


def lf_flux_all_edges(advec_var):
    '''
    Finds the LF flux at all the edges present in a mesh.
    '''
    element_lf_flux = af.np_to_af_array(np.zeros([advec_var.elements.shape[0],
                                                  4, params.N_LGL]))

    for element_0_tag in np.arange(advec_var.u_e_ij.shape[1]):
        for element_0_edge_id, element_1_tag in enumerate(
            advec_var.interelement_relations[element_0_tag]):
            if element_1_tag != -1:
                element_1_tag = advec_var.interelement_relations[element_0_tag,
                                                                 element_0_edge_id]

                element_1_edge_id = np.where(
                    advec_var.interelement_relations[element_1_tag] \
                    == element_0_tag)[0][0]

                u_element_0 = advec_var.u_e_ij[:, element_0_tag]
                u_element_1 = advec_var.u_e_ij[:, element_1_tag]

                u_element_0_at_edge = af.flat(u_at_edge(u_element_0,
                                                        element_0_edge_id))
                u_element_1_at_edge = af.flat(u_at_edge(u_element_1,
                                                        element_1_edge_id))
                
                element_lf_flux[element_0_tag,
                                element_0_edge_id] = lax_friedrichs_flux(
                                    u_element_0_at_edge,
                                    u_element_1_at_edge)

            if element_1_tag == -1:
                print('Element {} Tag at edge {} = -1'.format(element_0_tag,
                                                              element_0_edge_id))
                u_element_0 = advec_var.u_e_ij[:, element_0_tag]
                
                u_element_0_at_edge = af.flat(u_at_edge(u_element_0,
                                                        element_0_edge_id))
                u_element_1_at_edge = af.np_to_af_array(np.zeros([params.N_LGL]))
                
                element_lf_flux[element_0_tag,
                                element_0_edge_id] = lax_friedrichs_flux(
                                    u_element_0_at_edge,
                                    u_element_1_at_edge)
                
    print('Done')
    
    return element_lf_flux



def surface_term_vectorized(u, advec_var):
    '''
    '''
    # Testing lagrange interlolation
    surface_term = af.constant(0., d0 = params.N_LGL ** 2,
                               d1 = advec_var.elements.shape[0])

    dx_dxi  = 0.1
    dy_deta = 0.1
    
    element_lf_flux = lf_flux_all_edges(advec_var)
    
    for element_tag in np.arange(advec_var.elements.shape[0]):
        print('->', element_tag)
        for p in np.arange(params.N_LGL):
            for q in np.arange(params.N_LGL):
                
                index = p * params.N_LGL + q
                
                # Left Edge Integration
                edge_id     = 0

                xi_minus_1    = af.constant(-1., d0 = params.N_LGL, d1 = 1, dtype = af.Dtype.f64)
                Lp_xi_minus_1 = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[p], xi_minus_1))
                Fxi_minus_1   = af.flat(wave_equation_2d.F_x(element_lf_flux[element_tag, edge_id]))
                Lq_eta        = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[q], advec_var.eta_LGL))

                left_edge_integrand = xi_minus_1 * Lp_xi_minus_1 * Fxi_minus_1 * Lq_eta * dy_deta
                left_edge_integrand = lagrange.lagrange_interpolation_u(left_edge_integrand, advec_var)[:, :, 0]

                left_edge_integration = utils.integrate_1d(left_edge_integrand,
                                                        order = 9, scheme = 'gauss')

                # Bottom edge integration
                edge_id     = 1

                eta_minus_1    = af.constant(-1., d0 = params.N_LGL, d1 = 1, dtype = af.Dtype.f64)
                Lq_eta_minus_1 = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[q], eta_minus_1))
                Feta_minus_1   = af.flat(wave_equation_2d.F_y(element_lf_flux[element_tag, edge_id]))
                Lp_xi          = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[p], advec_var.xi_LGL))

                bottom_edge_integrand = eta_minus_1 * Lq_eta_minus_1 * Feta_minus_1 * Lp_xi * dx_dxi
                bottom_edge_integrand = lagrange.lagrange_interpolation_u(bottom_edge_integrand,
                                                                        advec_var)[:, :, 0]

                bottom_edge_integration = utils.integrate_1d(bottom_edge_integrand, order = 9, scheme = 'gauss')


                # Right Edge Integration

                edge_id     = 2

                xi_1    = af.constant(1., d0 = params.N_LGL, d1 = 1, dtype = af.Dtype.f64)
                Lp_xi_1 = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[p], xi_1))
                Fxi_1   = af.flat(wave_equation_2d.F_x(element_lf_flux[element_tag, edge_id]))
                Lq_eta  = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[q], advec_var.eta_LGL))

                right_edge_integrand = xi_1 * Lp_xi_1 * Fxi_1 * Lq_eta * dy_deta
                right_edge_integrand = lagrange.lagrange_interpolation_u(right_edge_integrand,
                                                                        advec_var)[:, :, 0]

                right_edge_integration = utils.integrate_1d(right_edge_integrand,
                                                            order = 9, scheme = 'gauss')

                # Top edge integration

                edge_id     = 3

                eta_1    = af.constant(1., d0 = params.N_LGL, d1 = 1, dtype = af.Dtype.f64)
                Lq_eta_1 = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[q], eta_1))
                Feta_1   = af.flat(wave_equation_2d.F_y(element_lf_flux[element_tag, edge_id]))
                Lp_xi    = af.flat(utils.polyval_1d(advec_var.lagrange_coeffs[p], advec_var.xi_LGL))

                top_edge_integrand = eta_1 * Lq_eta_1 * Feta_1 * Lp_xi * dx_dxi
                top_edge_integrand = lagrange.lagrange_interpolation_u(top_edge_integrand,
                                                                    advec_var)[:, :, 0]

                top_edge_integration = utils.integrate_1d(top_edge_integrand, order = 9, scheme = 'gauss')

                surface_term[index, element_tag] = -left_edge_integration - bottom_edge_integration + right_edge_integration + left_edge_integration

    #print(surface_term[:, 0])
    return surface_term


def b_vector(u, advec_var):
    '''
    '''
    b = volume_integral(u, advec_var) \
      - surface_term_vectorized(u, advec_var)

    return b

def RK4_timestepping(A_inverse, u, delta_t, gv):
    '''
    Implementing the Runge-Kutta (RK4) method to evolve the wave.

    Parameters
    ----------
    A_inverse : arrayfire.Array[N_LGL N_LGL 1 1]
                The inverse of the A matrix which was calculated
                using A_matrix() function.

    u         : arrayfire.Array[N_LGL N_Elements 1 1]
                u at the mapped LGL points

    delta_t   : float64
                The time-step by which u is to be evolved.

    Returns
    -------
    delta_u : arrayfire.Array [N_LGL N_Elements 1 1]
              The change in u at the mapped LGL points.
    '''

    k1 = af.matmul(A_inverse, b_vector(u, gv))
    k2 = af.matmul(A_inverse, b_vector(u + k1 * delta_t / 2, gv))
    k3 = af.matmul(A_inverse, b_vector(u + k2 * delta_t / 2, gv))
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t, gv))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u

def time_evolution(gv):
    '''
    '''
    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/2d_hdf5_%02d' %(int(params.N_LGL))
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    u         = gv.u_e_ij
    delta_t   = gv.delta_t_2d
    time      = gv.time_2d
    u_init    = gv.u_e_ij

    gauss_points    = gv.gauss_points
    gauss_weights   = gv.gauss_weights
    dLp_Lq          = gv.dLp_Lq
    dLq_Lp          = gv.dLq_Lp
    xi_LGL          = gv.xi_LGL
    lagrange_coeffs = gv.lagrange_coeffs
    Li_Lj_coeffs    = gv.Li_Lj_coeffs
    lobatto_weights = gv.lobatto_weights_quadrature

    A_inverse = af.np_to_af_array(np.linalg.inv(np.array(A_matrix(gv))))

    for i in trange(time.shape[0]):
        L1_norm = af.mean(af.abs(u_init - u))

        if (L1_norm >= 100):
            break
        if (i % 1) == 0:
            h5file = h5py.File('results/2d_hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(i)) + '.hdf5', 'w')
            dset   = h5file.create_dataset('u_i', data = u, dtype = 'd')

            dset[:, :] = u[:, :]


        u += +RK4_timestepping(A_inverse, u, delta_t, gv)


        #Implementing second order time-stepping.
        #u_n_plus_half =  u + af.matmul(A_inverse, b_vector(u))\
        #                      * delta_t / 2

        #u            +=  af.matmul(A_inverse, b_vector(u_n_plus_half))\
        #                  * delta_t

    return L1_norm

def u_analytical(t_n, gv):
    '''
    '''
    time = gv.delta_t_2d * t_n
    u_analytical_t_n = af.sin(2 * np.pi * (gv.x_e_ij - params.c_x * time) +
                              4 * np.pi * (gv.y_e_ij - params.c_y * time))

    return u_analytical_t_n
