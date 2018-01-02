#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import arrayfire as af

#af.set_backend('cpu')
#af.set_device(0)

from dg_maxwell import utils
from dg_maxwell import params


def LGL_points(N):
    '''
    Calculates : math: `N` Legendre-Gauss-Lobatto (LGL) points.
    LGL points are the roots of the polynomial 

    :math: `(1 - \\xi ** 2) P_{n - 1}'(\\xi) = 0`

    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.
    This function finds the roots of the above polynomial.

    Parameters
    ----------

    N : int
        Number of LGL nodes required
    
    Returns
    -------

    lgl : arrayfire.Array [N 1 1 1]
          The Lagrange-Gauss-Lobatto Nodes.
                          
    **See:** `document`_
    .. _document: https://goo.gl/KdG2Sv

    '''

    lgl_points = np.zeros(N)
    lgl_points[1:N-1] = (np.roots(np.polyder(sp.legendre(N-1))))
    lgl_points[0] = -1
    lgl_points[-1] = 1
    lgl_points.sort()
    lgl_points = af.np_to_af_array(lgl_points)
    
    return lgl_points


def Chebyshev_points(N):
    '''
    Calculates : math: `N` Chebyshev_points nodes.
    LGL points are the roots of the polynomial 

    :math: `(1 - \\xi ** 2) P_{n - 1}'(\\xi) = 0`

    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.
    This function finds the roots of the above polynomial.

    Parameters
    ----------

    N : int
        Number of LGL nodes required
    
    Returns
    -------

    lgl : arrayfire.Array [N 1 1 1]
          The Lagrange-Gauss-Lobatto Nodes.
                          
    **See:** `document`_
    .. _document: https://goo.gl/KdG2Sv

    '''
    chebyshev_points = []
    for k in range (1,(N+1)):
        chebyshev_points.append(np.cos(((2*k-1)/(2*N))*np.pi))
    chebyshev_points = np.asarray(chebyshev_points)
    chebyshev_points.sort()
    chebyshev_points         = af.np_to_af_array(chebyshev_points)

    return chebyshev_points

def lobatto_weights(n):
    '''
    Calculates and returns the weight function for an index n
    and points x.
    
    
    Parameters
    ----------
    n : int
        Lobatto weights for n quadrature points.
    
    
    Returns
    -------
    Lobatto_weights : arrayfire.Array
                      An array of lobatto weight functions for
                      the given x points and index.

    **See:** Gauss-Lobatto weights Wikipedia `link`_.
    
    .. _link: https://goo.gl/kYqTyK
    **Examples**
    
    lobatto_weight_function(4) returns the Gauss-Lobatto weights
    which are to be used with the Lobatto nodes 'LGL_points(4)'
    to integrate using Lobatto quadrature.

    '''
    xi_LGL = LGL_points(n)
    
    P = sp.legendre(n - 1)
    
    Lobatto_weights = (2 / (n * (n - 1)) / (P(xi_LGL))**2)
    Lobatto_weights = af.np_to_af_array(Lobatto_weights)
    
    return Lobatto_weights


def gauss_nodes(n):
    '''
    Calculates :math: `N` Gaussian nodes used for Integration by
    Gaussia quadrature.
    Gaussian node :math: `x_i` is the `i^{th}` root of
    :math: `P_n(\\xi)`
    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.

    Parameters
    ----------

    n : int
        The number of Gaussian nodes required.

    Returns
    -------

    gauss_nodes : numpy.ndarray
                  The Gauss nodes :math: `x_i`.

    **See:** A Wikipedia article about the Gauss-Legendre quadrature `here`_
    
    .. _here: https://goo.gl/9gqLpe

    '''
    gauss_nodes = np.polynomial.legendre.leggauss(n)[0]

    
    return gauss_nodes


def gaussian_weights(N):
    '''
    Returns the gaussian weights :math:`w_i` for :math:`N` Gaussian Nodes
    at index :math:`i`. They are given by

    .. math:: w_i = \\frac{2}{(1 - x_i^2) P'n(x_i)}

    Where :math:`x_i` are the Gaussian nodes and :math:`P_{n}(\\xi)`
    are the Legendre polynomials.
    
    Parameters
    ----------
    
    N : int
        Number of Gaussian nodes for which the weight is to be calculated.
            
   
    Returns
    -------
    
    gaussian_weight : arrayfire.Array [N_quad 1 1 1]
                      The gaussian weights.
    '''
    
    gaussian_weight = np.polynomial.legendre.leggauss(N)[1]
    gaussian_weight = af.np_to_af_array(gaussian_weight)
    
    return gaussian_weight


def lagrange_polynomials(x):    
    '''
    A function to get the analytical form and the coefficients of
    Lagrange basis polynomials evaluated using x nodes.
    
    It calculates the Lagrange basis polynomials using the formula:
    
    .. math:: \\
        L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}

    Parameters
    ----------
    
    x : numpy.array [N_LGL 1 1 1]
        Contains the :math: `x` nodes using which the
        lagrange basis functions need to be evaluated.

    Returns
    -------
    
    lagrange_basis_poly   : list
                            A list of size `x.shape[0]` containing the
                            analytical form of the Lagrange basis polynomials
                            in numpy.poly1d form. This list is used in
                            integrate() function which requires the analytical
                            form of the integrand.

    lagrange_basis_coeffs : numpy.ndarray
                            A :math: `N \\times N` matrix containing the
                            coefficients of the Lagrange basis polynomials such
                            that :math:`i^{th}` lagrange polynomial will be the
                            :math:`i^{th}` row of the matrix.

    **Examples**
    
    lagrange_polynomials(4)[0] gives the lagrange polynomials obtained using
    4 LGL points in poly1d form

    lagrange_polynomials(4)[0][2] is :math: `L_2(\\xi)`
    lagrange_polynomials(4)[1] gives the coefficients of the above mentioned
    lagrange basis polynomials in a 2D array.

    lagrange_polynomials(4)[1][2] gives the coefficients of :math:`L_2(\\xi)`
    in the form [a^2_3, a^2_2, a^2_1, a^2_0]

    '''
 
    X = np.array(x)
    lagrange_basis_poly   = []
    lagrange_basis_coeffs = np.zeros([X.shape[0], X.shape[0]])
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_j = np.ones([1])
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_j = polymult(lagrange_basis_j, np.asarray([1, -X[m]],dtype = np.float64) \
                                    / ((X[j] - X[m])))

        lagrange_basis_coeffs[j] = lagrange_basis_j
        lagrange_basis_poly.append(lagrange_basis_j)        
    
    return lagrange_basis_poly, lagrange_basis_coeffs


def polymult(arr_1, arr_2):
    arr_3 = np.zeros(arr_1.size + arr_2.size-1)
    for i in range(0,arr_1.size):
        for j in range(0,arr_2.size):
            arr_3[i+j] += arr_1[i]*arr_2[j]
    return arr_3



def lagrange_function_value(lagrange_coeff_array):
    '''

    Funtion to calculate the value of lagrange basis functions over LGL
    nodes.

    Parameters
    ----------
    
    lagrange_coeff_array : arrayfire.Array[N_LGL N_LGL 1 1]
                           Contains the coefficients of the
                           Lagrange basis polynomials
    
    Returns
    -------
    
    L_i : arrayfire.Array [N 1 1 1]
          The value of lagrange basis functions calculated over the LGL
          nodes.

    **Examples**
    
    lagrange_function_value(4) gives the value of the four
    Lagrange basis functions evaluated over 4 LGL points
    arranged in a 2D array where Lagrange polynomials
    evaluated at the same LGL point are in the same column.
    
    Also the value lagrange basis functions at LGL points has the property,
    
    L_i(xi_k) = 0 for i != k
              = 1 for i  = k
    
    It follows then that lagrange_function_value returns an identity matrix.
    
    '''
    xi_tile    = af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL))
    power      = af.flip(af.range(params.N_LGL))
    power_tile = af.tile(power, 1, params.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(params.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array[index], xi_pow)
    
    return L_i



def integrate(integrand_coeffs):
    '''
    Performs integration according to the given quadrature method
    by taking in the coefficients of the polynomial and the number of
    quadrature points.
    The number of quadrature points and the quadrature scheme are set
    in params.py module.
    
    Parameters
    ----------
    
    integrand_coeffs : arrayfire.Array [M N 1 1]
                       The coefficients of M number of polynomials of order N
                       arranged in a 2D array.
    Returns
    -------
    
    Integral : arrayfire.Array [M 1 1 1]
               The value of the definite integration performed using the
               specified quadrature method for M polynomials.
    '''


    integrand      = integrand_coeffs

    if (params.scheme == 'gauss_quadrature'):
        #print('gauss_quad')

        gaussian_nodes = params.gauss_points
        Gauss_weights  = params.gauss_weights

        nodes_tile   = af.transpose(af.tile(gaussian_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_gauss_nodes = af.matmul(integrand, nodes_weight)
        integral             = af.sum(value_at_gauss_nodes, 1)
 
    if (params.scheme == 'lobatto_quadrature'):
        #print('lob_quad')

        lobatto_nodes   = params.lobatto_quadrature_nodes
        Lobatto_weights = params.lobatto_weights_quadrature

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile


        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        integral               = af.sum(value_at_lobatto_nodes, 1)


    return integral




def lagrange_interpolation_u(u):
    '''

    Calculates the coefficients of the Lagrange interpolation using
    the value of u at the mapped LGL points in the domain.

    The interpolation using the Lagrange basis polynomials is given by

    :math:`L_i(\\xi) u_i(\\xi)`

    Where L_i are the Lagrange basis polynomials and u_i is the value
    of u at the LGL points.

    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        The value of u at the mapped LGL points.

    Returns
    -------
    lagrange_interpolated_coeffs : arrayfire.Array[1 N_LGL N_Elements 1]
                                   The coefficients of the polynomials obtained
                                   by Lagrange interpolation. Each polynomial
                                   is of order N_LGL - 1.

    '''
    lagrange_coeffs_tile = af.tile(params.lagrange_coeffs, 1, 1,\
                                               params.N_Elements)
    reordered_u          = af.reorder(u, 0, 2, 1)

    lagrange_interpolated_coeffs = af.sum(af.broadcast(utils.multiply,\
                                             reordered_u, lagrange_coeffs_tile), 0)

    return lagrange_interpolated_coeffs


def L1_norm(u):
    '''
    A function to calculate the L1 norm of error using
    the polynomial obtained using Lagrange interpolation

    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        Difference between analytical and numerical u at the mapped LGL points.

    Returns
    -------
    L1_norm : float64
              The L1 norm of error.

    '''
    interpolated_coeffs = af.reorder(lagrange_interpolation_u(\
                                           u), 2, 1, 0)

    L1_norm = af.sum(integrate(interpolated_coeffs))

    return L1_norm


def lagrange_interpolation(fn_i):
    '''
    Finds the general interpolation of a function.
    
    Parameters
    ----------
    fn_i : af.Array [N N_LGL 1 1]
           Value of :math:`N` functions at the LGL points.
    
    Returns
    -------
    lagrange_interpolation : af.Array [N N_LGL 1 1]
                             :math:`N` interpolated polynomials for
                             :math:`N` functions.
    '''
    
    fn_i = af.transpose(af.reorder(fn_i, d0 = 2, d1 = 1, d2 = 0))
    lagrange_interpolation = af.broadcast(utils.multiply,
                                          params.lagrange_coeffs, fn_i)
    lagrange_interpolation = af.reorder(af.sum(lagrange_interpolation, dim = 0),
                                        d0 = 2, d1 = 1, d2 = 0)

    return lagrange_interpolation


def eval_lagrange_basis(point_arr, j):
    
    '''
    Finds the value of an array at
    
    Parameters
    -------------
    point_arr : numpy_array[N]
                Contains the points at which you want to evaluate the basis value
    j         : the value of the order of lagrange basis you want to evaluate
    
    Returns
    -------
    lagrange_interpolation : numpy_array[N]
                             Contains the evaluated values
    '''
    xi_lgl = np.asarray(params.xi_LGL)
    weight_arr = np.asarray(params.weight_arr)
    eval_point_arr = np.zeros(point_arr.size)
    for k in range(0, point_arr.size):
        flag = False
        for l in range(0, xi_lgl.size):
            if(point_arr[k] == xi_lgl[l]):
                if(j==l):
                    eval_point_arr[k] = 1
                    flag = True
                    break
                flag = True
                eval_point_arr[k] = 0
        if(flag == True):
            continue
        denom = 0
        for i in range(0, weight_arr.size):
            denom += weight_arr[i]/(point_arr[k]-xi_lgl[i])
        numer = weight_arr[j]/(point_arr[k]-xi_lgl[j])
        eval_point_arr[k] = numer/denom
    return eval_point_arr

def weight_arr_fun(xi_lgl):
    '''
    Finds the eights in the barycentric formulation
    
    Parameters
    -------------
    xi_lgl                 : lgl points    
    
    Returns
    -------
    weight_arr             : array containing weights
    
    '''
    xi_lgl = params.xi_LGL
    xi_lgl = np.asarray(xi_lgl)
    weight_arr = np.zeros(xi_lgl.size, dtype = np.float64)
    for j in range(0, xi_lgl.size):
        weight = 1
        for i in range(0, xi_lgl.size):
            if(i!=j):
                weight *= 1/(xi_lgl[j]-xi_lgl[i])
        weight_arr[j] = weight
    return weight_arr


def eval_diff_lagrange_basis(lobatto_nodes, j):
    '''
    Finds the value of derivative of lagrange basis polynomials 
    at the lobatto nodes
    
    Parameters
    -------------
    lobatto_nodes                 : lgl points    
    j                             : order of the lagrange basis polynomial
    
    Returns
    -------
    eval_lobatto_nodes_diff       : array contining the value of derivative of lagrange
                                    basis polynomial j
    
    '''
    xi_lgl = np.asarray(params.xi_LGL)
    weight_arr = np.asarray(params.weight_arr)
    eval_lobatto_nodes_diff = np.zeros(xi_lgl.size)
    for k in range(0, xi_lgl.size):
        if(j == k):
            for m in range(0, xi_lgl.size):
                if(m != k):
                    eval_lobatto_nodes_diff[k] += 1/(xi_lgl[k]-xi_lgl[m])
        else:
            eval_lobatto_nodes_diff[k] = weight_arr[j]/(weight_arr[k]*(xi_lgl[k]-xi_lgl[j]))
    return eval_lobatto_nodes_diff


def b_matrix_eval():
        '''
    Finds the matrix helpful in calculating the volume_term 
    by the barycentric formulation
    
    
    Returns
    -------
    b_matrix       : matrix specified in the calculation of the b_matrix in 
                     balavaruns's pdf.
    
    '''
    
    
    xi_lgl = np.asarray(params.xi_LGL)
    weight_arr = weight_arr_fun(xi_lgl)
    lobatto_nodes   = params.lobatto_quadrature_nodes
    lobatto_nodes   = np.asarray(lobatto_nodes)
    Lobatto_weights = params.lobatto_weights_quadrature
    Lobatto_weights = np.asarray(Lobatto_weights)
    b_matrix = np.zeros((xi_lgl.size, xi_lgl.size))
    for i in range(0, xi_lgl.size):
        temp_arr = eval_diff_lagrange_basis(lobatto_nodes, i)
        for j in range(0, xi_lgl.size):
            b_matrix[i][j] = Lobatto_weights[j] * temp_arr[j]
    
    return b_matrix
    

def eval_arr(point_arr, function_arr):
    '''
    Finds the interpolation of the point_arr when the value at lobatto_nodes
    is given in the function_arr
    
    Parameters
    -------------
    point_arr                : points at which you want to calculate the interpolated value
    function_arr             : value of the function at the lobatto points
    
    Returns
    -------
    point_eval_arr           : returns the evaluated interpolated array
    
    '''
    xi_lgl = np.asarray(params.xi_LGL)
    weight_arr = weight_arr_fun(xi_lgl)
    point_eval_arr = np.zeros(point_arr.size, dtype = np.float64)
    for j in range(0, point_arr.size):
        flag = False
        for l in range(0, xi_lgl.size):
            if(point_arr[j] == xi_lgl[l]):
                print('Evaluated points matches the actual point')
                point_eval_arr[j] = function_arr[j]
                flag = True
                break
        if(flag == True):
            continue
        denom = 0
        for i in range(0, weight_arr.size):
            denom += weight_arr[i]/(point_arr[j] - xi_lgl[i])
        numer = 0
        for i in range(0, function_arr.size):
            numer += (weight_arr[i]/(point_arr[j] - xi_lgl[i]))*(function_arr[i])
        point_eval_arr[j] = numer/denom
        
    return point_eval_arr
