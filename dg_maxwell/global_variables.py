#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import utils
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d
from dg_maxwell import msh_parser
from dg_maxwell import isoparam
from dg_maxwell import advection_2d_arbit_mesh as advec_2d__arbit_mesh

af.set_backend(params.backend)
af.set_device(params.device)

def function(x):
    return 0.07 * np.sin(2 * np.pi * x)


class advection_variables:
    '''
    Stores and initializes such variables which are called repeatedly
    in different functions.
    '''
    
    def __init__(self, N_LGL, N_quad, x_nodes,
                 N_elements, c, total_time, wave,
                 c_x, c_y, courant, mesh_file,
                 total_time_2d):
        '''
        Initializes the variables using the user parameters.
        
        Parameters
        ----------
        N_LGL : int
                Number of LGL points(for both :math:`2D` and :math:`1D` wave
                equation solver).
        N_quad : int
                 Number of the quadrature points to use in Gauss-Lobatto or
                 Gauss-Legendre quadrature.
        x_nodes : af.Array [2 1 1 1]
                  :math:`x` nodes for the :math:`1D` wave equation elements.
                  
        N_elements : int
                     Number of elements in a :math:`1D` domain.
                     
        c : float64
            Wave speed for 1D wave equation.
            
        total_time : float64
                     Total time for which :math:`1D` wave equation is to be
                     evolved.
                     
        wave : str
               Used to set u_init to ``sin`` or ``cos``.
               
        c_x : float64
              :math:`x` component of wave speed for a :math:`2D` wave.
              
        c_y : float64
              :math:`y` component of wave speed for a :math:`2D` wave.
              
        courant : float64
                  Courant parameter used for the time evolution of the wave.
                  
        mesh_file : str
                    Path of the mesh file for the 2D wave equation.
                    
        total_time_2d : float64
                        Total time for which the wave is to propogated.
        
        Returns
        -------
        None
        '''

        self.xi_LGL  = lagrange.LGL_points(N_LGL)
        self.eta_LGL = lagrange.LGL_points(N_LGL)

        # N_Gauss number of Gauss nodes.
        self.gauss_points = af.np_to_af_array(lagrange.gauss_nodes(N_quad))

        # The Gaussian weights.
        self.gauss_weights = lagrange.gaussian_weights(N_quad)

        # The lobatto nodes to be used for integration.
        self.lobatto_quadrature_nodes = lagrange.LGL_points(N_quad)

        # The lobatto weights to be used for integration.
        self.lobatto_weights_quadrature = lagrange.lobatto_weights(N_quad)

        # An array containing the coefficients of the lagrange basis polynomials.
        self.lagrange_coeffs = lagrange.lagrange_polynomial_coeffs(self.xi_LGL)

        self.lagrange_basis_value = lagrange.lagrange_function_value(
            self.lagrange_coeffs, self.xi_LGL)

        self.diff_pow = af.flip(af.transpose(af.range(N_LGL - 1) + 1), 1)
        self.dl_dxi_coeffs = af.broadcast(utils.multiply, self.lagrange_coeffs[:, :-1],
                                          self.diff_pow)

        self.element_size    = af.sum((x_nodes[1] - x_nodes[0]) / N_elements)
        self.elements_xi_LGL = af.constant(0, N_elements, N_LGL)
        self.elements        = utils.linspace(af.sum(x_nodes[0]),
                                              af.sum(x_nodes[1]
                                                     - self.element_size),
                                              N_elements)

        self.np_element_array = np.concatenate((
            af.transpose(self.elements),
            af.transpose(self.elements + self.element_size)))
        
        self.element_mesh_nodes = utils.linspace(af.sum(x_nodes[0]),
                                                 af.sum(x_nodes[1]),
                                                 N_elements + 1)

        self.element_array = af.transpose(
            af.interop.np_to_af_array(self.np_element_array))
        self.element_LGL   = wave_equation.mapping_xi_to_x(
            af.transpose(self.element_array), self.xi_LGL)

        # The minimum distance between 2 mapped LGL points.
        self.delta_x = af.min((self.element_LGL - af.shift(self.element_LGL,
                                                           1, 0))[1:, :])

        # dx_dxi for elements of equal size.
        self.dx_dxi  = af.mean(wave_equation.dx_dxi_numerical(
            self.element_mesh_nodes[0 : 2],
            self.xi_LGL))

        # The value of time-step.
        self.delta_t = self.delta_x / (4 * c)

        # Array of timesteps seperated by delta_t.
        self.time = utils.linspace(0, int(total_time / self.delta_t)
                                   * self.delta_t,
                                   int(total_time / self.delta_t))

        # Initializing the amplitudes. Change u_init to required initial conditions.
        if (wave=='sin'):
            self.u_init = af.sin(2 * np.pi * self.element_LGL)

        if (wave=='gaussian'):
            self.u_init = np.e ** (-(self.element_LGL) ** 2 / 0.4 ** 2)

        self.test_array = af.np_to_af_array(np.array(self.u_init))



        # The parameters below are for 2D advection
        # -----------------------------------------


        ########################################################################
        #######################2D Wave Equation#################################
        ########################################################################


        self.xi_i  = af.flat(af.transpose(af.tile(self.xi_LGL, 1, N_LGL)))
        self.eta_j = af.tile(self.xi_LGL, N_LGL)

        self.dLp_xi_ij = af.moddims(af.reorder(af.tile(utils.polyval_1d(
            self.dl_dxi_coeffs, self.xi_i), 1, 1, N_LGL),
        1, 2, 0), N_LGL ** 2, 1, N_LGL ** 2)
        self.Lp_xi_ij  = af.moddims(af.reorder(af.tile(utils.polyval_1d(
            self.lagrange_coeffs, self.xi_i),
        1, 1, N_LGL), 1, 2, 0), N_LGL ** 2, 1, N_LGL ** 2)

        self.dLq_eta_ij = af.tile(af.reorder(utils.polyval_1d(
            self.dl_dxi_coeffs,
            self.eta_j),
        1, 2, 0), 1, 1, N_LGL)

        self.Lq_eta_ij  = af.tile(af.reorder(utils.polyval_1d(
            self.lagrange_coeffs, self.eta_j), 1, 2, 0), 1, 1, N_LGL)

        self.dLp_Lq = self.Lq_eta_ij * self.dLp_xi_ij
        self.dLq_Lp = self.Lp_xi_ij  * self.dLq_eta_ij
        
        self.Li_Lj_coeffs = wave_equation_2d.Li_Lj_coeffs(N_LGL)

        self.delta_y = self.delta_x

        self.delta_t_2d = courant * self.delta_x * self.delta_y \
                        / (self.delta_x * c_x + self.delta_y * c_y)

        self.c_lax_2d_x = c_x
        self.c_lax_2d_y = c_y

        self.nodes, self.elements = msh_parser.read_order_2_msh(mesh_file)
        
        #############################################################
        ## Code to distort the elements by a function
        #############################################################
        #new_nodes = self.nodes.copy()

        #for node_tag, node in enumerate(self.nodes):
            #if np.all((node == [-1, 1])) or np.all(node == [-1, -1]) or np.all(node == [1, -1]) or np.all(node == [1, 1]):
                #print(node, '\t vertex')
                #continue
            #if ((node[1] != -1) and (node[1] != 1)):
                #new_nodes[node_tag, 1] += function(node[0])
            
            #if ((node[0] != -1) and (node[0] != 1)):
                #new_nodes[node_tag, 0] += function(node[1])
        #self.nodes = new_nodes
        #############################################################
        
        
        #############################################################
        ## Element reordering for 4x4 non-contiguous mesh
        ## For [TESTING], remove after fixing the bug.
        #############################################################
        #elements_reorder_nodes = self.elements[:, :-1].copy()
        
        #for element_tag in np.arange(8, 12):
            #elements_reorder_nodes[element_tag] = \
                #np.roll(elements_reorder_nodes[element_tag], 2)

        #for element_tag in np.arange(12, 16):
            #elements_reorder_nodes[element_tag] = \
                #np.flip(elements_reorder_nodes[element_tag], axis = 0)
            #elements_reorder_nodes[element_tag] = \
                #np.roll(elements_reorder_nodes[element_tag], 3)

        #self.elements[:, :-1] = elements_reorder_nodes
        #############################################################
        #############################################################
        
        self.x_e_ij = af.np_to_af_array(np.zeros([N_LGL * N_LGL,
                                                  len(self.elements)]))
        self.y_e_ij = af.np_to_af_array(np.zeros([N_LGL * N_LGL,
                                                  len(self.elements)]))

        for element_tag, element in enumerate(self.elements):
            self.x_e_ij[:, element_tag] = isoparam.isoparam_x_2D(
                self.nodes[element, 0], self.xi_i, self.eta_j)
            self.y_e_ij[:, element_tag] = isoparam.isoparam_y_2D(
                self.nodes[element, 1], self.xi_i, self.eta_j)

        self.u_e_ij = np.e**(-((self.x_e_ij - (0)) ** 2 + self.y_e_ij ** 2) / 0.4**2)
        #af.sin(self.x_e_ij * 2 * np.pi + self.y_e_ij * 4 * np.pi)

        # Array of timesteps seperated by delta_t.
        self.time_2d = utils.linspace(0, int(total_time_2d / self.delta_t_2d)
                                      * self.delta_t_2d,
                                      int(total_time_2d / self.delta_t_2d))

        # Applying the periodic boundary conditions
        self.interelement_relations = af.np_to_af_array(
            msh_parser.interelement_relations(self.elements))
        
        ###Code to apply periodic BC for square 10x10 contiguous case.
        #for vertical_boundary in params.vertical_boundary_elements_pbc:
            #self.interelement_relations[vertical_boundary[0],
                                        #0] = vertical_boundary[1]
            #self.interelement_relations[vertical_boundary[1],
                                        #2] = vertical_boundary[0]

        #for horizontal_boundary in params.horizontal_boundary_elements_pbc:
            #if horizontal_boundary[1] < 110:
                #self.interelement_relations[horizontal_boundary[0],
                                            #3] = horizontal_boundary[1]
                #self.interelement_relations[horizontal_boundary[1],
                                            #1] = horizontal_boundary[0]
                
            #else:
                #self.interelement_relations[horizontal_boundary[0],
                                            #3] = horizontal_boundary[1]
                #self.interelement_relations[horizontal_boundary[1],
                                            #3] = horizontal_boundary[0]
        
        ################################################################
        ###Code to apply periodic BC for square 4x4 non-contiguous case.
        ################################################################
        #for vertical_boundary in params.vertical_boundary_elements_pbc:
            #self.interelement_relations[vertical_boundary[0],
                                        #0] = vertical_boundary[1]
            #self.interelement_relations[vertical_boundary[1],
                                        #2] = vertical_boundary[0]

        #for horizontal_boundary in params.horizontal_boundary_elements_pbc:
            #self.interelement_relations[horizontal_boundary[0], 3] = \
                #horizontal_boundary[1]
            #self.interelement_relations[horizontal_boundary[1], 1] = \
                #horizontal_boundary[0]

        ### Code to apply BC for 4x4 contiguous case
        ## Vertical boundary conditions
        #self.interelement_relations[0, 0] = 3
        #self.interelement_relations[3, 2] = 0

        #self.interelement_relations[4, 0] = 7
        #self.interelement_relations[7, 2] = 4

        #self.interelement_relations[8, 0] = 11
        #self.interelement_relations[11, 2] = 8

        #self.interelement_relations[12, 0] = 15
        #self.interelement_relations[15, 2] = 12


        ## Horizontal boundary conditions
        #self.interelement_relations[0, 3] = 12
        #self.interelement_relations[12, 1] = 0

        #self.interelement_relations[1, 3] = 13
        #self.interelement_relations[13, 1] = 1

        #self.interelement_relations[2, 3] = 14
        #self.interelement_relations[14, 1] = 2

        #self.interelement_relations[3, 3] = 15
        #self.interelement_relations[15, 1] = 3

        # Code to apply periodic BC for square 4x4 non-contiguous case.

        #self.interelement_relations[0, 0] = 15
        #self.interelement_relations[15, 2] = 0

        #self.interelement_relations[2, 0] = 13
        #self.interelement_relations[13, 2] = 2

        #self.interelement_relations[4, 0] = 11
        #self.interelement_relations[11, 1] = 4

        #self.interelement_relations[6, 0] = 10
        #self.interelement_relations[10, 1] = 6

        #self.interelement_relations[0, 3] = 6
        #self.interelement_relations[6, 1] = 0

        #self.interelement_relations[1, 3] = 7
        #self.interelement_relations[7, 1] = 1

        #self.interelement_relations[14, 1] = 8
        #self.interelement_relations[8, 0] = 14

        #self.interelement_relations[15, 1] = 10
        #self.interelement_relations[10, 0] = 15

        print('advection_variables __init__ completed')
        
        return
    
