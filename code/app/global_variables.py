#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from scipy import special as sp

from app import lagrange
from app import wave_equation
from utils import utils

LGL_list = [ \
[-1.0,1.0],                                                               \
[-1.0,0.0,1.0],                                                           \
[-1.0,-0.4472135955,0.4472135955,1.0],                                    \
[-1.0,-0.654653670708,0.0,0.654653670708,1.0],                            \
[-1.0,-0.765055323929,-0.285231516481,0.285231516481,0.765055323929,1.0], \
[-1.0,-0.830223896279,-0.468848793471,0.0,0.468848793471,0.830223896279,  \
1.0],                                                                     \
[-1.0,-0.87174014851,-0.591700181433,-0.209299217902,0.209299217902,      \
0.591700181433,0.87174014851,1.0],                                        \
[-1.0,-0.899757995411,-0.677186279511,-0.363117463826,0.0,0.363117463826, \
0.677186279511,0.899757995411,1.0],                                       \
[-1.0,-0.919533908167,-0.738773865105,-0.47792494981,-0.165278957666,     \
0.165278957666,0.47792494981,0.738773865106,0.919533908166,1.0],          \
[-1.0,-0.934001430408,-0.784483473663,-0.565235326996,-0.295758135587,    \
0.0,0.295758135587,0.565235326996,0.784483473663,0.934001430408,1.0],     \
[-1.0,-0.944899272223,-0.819279321644,-0.632876153032,-0.399530940965,    \
-0.136552932855,0.136552932855,0.399530940965,0.632876153032,             \
0.819279321644,0.944899272223,1.0],                                       \
[-1.0,-0.953309846642,-0.846347564652,-0.686188469082,-0.482909821091,    \
-0.249286930106,0.0,0.249286930106,0.482909821091,0.686188469082,         \
0.846347564652,0.953309846642,1.0],                                       \
[-0.999999999996,-0.959935045274,-0.867801053826,-0.728868599093,         \
-0.550639402928,-0.342724013343,-0.116331868884,0.116331868884,           \
0.342724013343,0.550639402929,0.728868599091,0.86780105383,               \
0.959935045267,1.0],                                                      \
[-0.999999999996,-0.965245926511,-0.885082044219,-0.763519689953,         \
-0.60625320547,-0.420638054714,-0.215353955364,0.0,0.215353955364,        \
0.420638054714,0.60625320547,0.763519689952,0.885082044223,               \
0.965245926503,1.0],                                                      \
[-0.999999999984,-0.9695680463,-0.899200533072,-0.792008291871,           \
-0.65238870288,-0.486059421887,-0.299830468901,-0.101326273522,           \
0.101326273522,0.299830468901,0.486059421887,0.652388702882,              \
0.792008291863,0.899200533092,0.969568046272,0.999999999999]]


for idx in np.arange(len(LGL_list)):
    LGL_list[idx] = np.array(LGL_list[idx], dtype = np.float64)
    LGL_list[idx] = af.interop.np_to_af_array(LGL_list[idx])

x_nodes         = af.interop.np_to_af_array(np.array([-1., 1.]))
N_LGL           = 16
xi_LGL          = None # N_LGL points
lBasisArray     = None # Lagrange basis array calculated using xi_LGL points
lobatto_weights = None # Lobatto weights to be used in Lobatto quadrature
N_Elements      = None # Number of elements in which the domain is to be divided
element_LGL     = None # x coordinates corresponding to the xi coordinates
                       # for each element
u               = None # value of the wave at all the x points corresponding to
                       # the LGL nodes.
time            = None
c               = None # Wave speed
dLp_xi          = None
c_lax           = None # Lax friedrichs flux
delta_t         = None

def populateGlobalVariables(Number_of_LGL_pts = 8,
                            Number_of_elements = 10,
                            c_ = 4.,
                            c_lax_ = 4.,
                            total_time_ = 4):
    '''
    For doing the time evolution of the wave function we
    need many constant variables throughout the program.
    These variables have been declared as a global variables in our program.
    The following are the list and explaination of the global variables which
    we are using in our program.
    
    x_nodes : af.array [2 1 1 1]
              This function stores the domain of the wave. By default it is
              set to [-1, 1] in the program. This means that the domain is from
              :math:`x \\epsilon [-1., 1.]`. This domain will be split
              into `Number_of_elements` elements.
    
    N_Elements : int
                 The number of elements in which the domain is to be divided.
                 
    N_LGL   : int
              The number of `LGL` nodes in an element.
              
    xi_LGL  : af.array [N_LGL 1 1 1]
              `N_LGL` LGL points.
    
    element_LGL : af.array [N_LGL N_Elements 1 1]
                  The domain is divided into `N_elements` number of elements
                  and each element has N_LGL points corresponding to the
                  each LGL point.
                  This variable stores the :math:`x` coordinates corresponding
                  to each `xi_LGL` for each element.
                  element_LGL[n_LGL, n_element] returns the :math:`n\\_LGL^{th}`
                  :math:`x` coordinate for the :math:`n_element^{th}` element.
                  
    lBasisArray : af.array [N_LGL N_LGL 1 1]
                  Contains the coefficients of the Lagrange basis functions
                  created using LGL nodes stored in xi_LGL.
                  lBasisArray[i, j] is the coefficient corresponding to the
                  :math:`(N_LGL - j - 1)^{th}` power of :math:`x` in the
                  :mat:`i^th` Lagrange polynomial.
    
    dLp_xi      : af.array [N_LGL N_LGL 1 1]
                  Stores the value of the derivative :math:`\\frac{dL_p}{d\\xi}`
                  at all the LGL points.
    
    lobatto_weights : af.array [N_LGL 1 1 1]
                      Lobatto weights to integrate using Gauss-Lobatto
                      quadrature.
                      
    c : float
        Number denoting the wave speed.
        
    c_lax : float
            A number used in the calculation of the Lax–Friedrichs flux.
            
    delta_t : float
              :math:`dt` between each time evolution.
    
    time    : af.array
              Array containing the time for which the :math:`u` is to be
              calculated.
    
    u : af.array [N_LGL N_Elements time.shape[0] 1]
        Stores :math:`u` calculated at the :math:`x` cordinates stored in
        the variable `element_LGL` at time :math:`t` stored in stored in
        variable `time`.
        
        
    Parameters
    ----------
    Number_of_LGL_pts : int
                        Number of LGL nodes.

    Number_of_elements : int
                         Number of elements into which the domain is to be
                         split.
    c_ : float
         Number denoting the wave speed.
        
    c_lax_ : float
             A number used in the calculation of the Lax–Friedrichs flux.

    total_time_ : float
                  Total time for which the simulation will find the time
                  evolution of the wave.
    '''

    global N_LGL
    global xi_LGL
    global lBasisArray
    global lobatto_weights
    N_LGL       = Number_of_LGL_pts
    xi_LGL      = lagrange.LGL_points(N_LGL)
    lBasisArray = af.interop.np_to_af_array( \
        lagrange.lagrange_basis_coeffs(xi_LGL))

    lobatto_weights = af.interop.np_to_af_array(\
        lobatto_weight_function(N_LGL, xi_LGL))

    global N_Elements
    global element_LGL
    global elementMeshNodes

    N_Elements       = Number_of_elements
    element_size     = af.sum((x_nodes[1] - x_nodes[0]) / N_Elements)
    elements_xi_LGL  = af.constant(0, N_Elements, N_LGL)
    elements         = utils.linspace(af.sum(x_nodes[0]),
                                        af.sum(x_nodes[1] - element_size),
                                        N_Elements)
    np_element_array = np.concatenate((af.transpose(elements), 
                            af.transpose(elements + element_size)))
    

    elementMeshNodes = utils.linspace(af.sum(x_nodes[0]),
                                        af.sum(x_nodes[1]),
                                        N_Elements + 1)

    
    element_array = af.transpose(af.interop.np_to_af_array(np_element_array))
    element_LGL = wave_equation.mappingXiToX(\
                                        af.transpose(element_array), xi_LGL)

    global c
    global c_lax
    global delta_t
    c       = c_
    delta_x = af.min((element_LGL - af.shift(element_LGL, 1, 0))[1:, :])
    delta_t = delta_x / (20. * c)
    c_lax   = c_lax_  # Was previously taken to be 0.1 even works if it's
                 # taken to be c.


    global u
    global time
    total_time = total_time_
    no_of_steps = int(total_time / delta_t)
    end_time    = delta_t * no_of_steps
    time        = utils.linspace(0, end_time, no_of_steps)
    time       = utils.linspace(0, end_time, no_of_steps)
    u_init     = np.e ** (-(element_LGL) ** 2 / 0.2 ** 2)
    #u_init     = af.np_to_af_array((np.cos(np.pi * element_LGL / 2))**2)
    u          = af.constant(0, N_LGL, N_Elements,
                             time.shape[0],
                             dtype = af.Dtype.f64)

    u[:, :, 0] = u_init

    global dLp_xi
    dLp_xi = dLp_xi_LGL()

    return


def lobatto_weight_function(n, x):
    '''
    Calculates and returns the weight function for an index :math:`n`
    and points :math: `x`.

    :math::
        `w_{n} = \\frac{2 P(x)^2}{n (n - 1)}`,
        Where P(x) is $ (n - 1)^th $ index.

    Parameters
    ----------
    n : int
        Index for which lobatto weight function

    x : arrayfire.Array
        1D array of points where weight function is to be calculated.

    .. lobatto weight function -
    https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules

    Returns
    -------
    (2 / (n * (n - 1)) / (P(x))**2) : arrayfire.Array
                                        An array of lobatto weight functions for
                                        the given :math: `x` points and index.

    '''
    P = sp.legendre(n - 1)

    return (2 / (n * (n - 1)) / (P(x))**2)


def lagrange_basis_function():
    '''
    Funtion which calculates the value of lagrange basis functions over LGL
    nodes.

    Returns
    -------
    L_i    : arrayfire.Array [N 1 1 1]
                The value of lagrange basis functions calculated over the LGL
                nodes.
    '''
    xi_tile    = af.transpose(af.tile(xi_LGL, 1, N_LGL))
    power      = af.flip(af.range(N_LGL))
    power_tile = af.tile(power, 1, N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(N_LGL)
    L_i        = af.blas.matmul(lBasisArray[index], xi_pow)

    return L_i


def dLp_xi_LGL():
    '''
    Function to calculate :math: `\\frac{d L_p(\\xi_{LGL})}{d\\xi}`
    as a 2D array of :math: `L_i (\\xi_{LGL})`. Where i varies along rows
    and the nodes vary along the columns.

    Returns
    -------
    dLp_xi        : arrayfire.Array [N N 1 1]
                    A 2D array :math: `L_i (\\xi_p)`, where i varies
                    along dimension 1 and p varies along second dimension.
    '''
    differentiation_coeffs = (af.transpose(af.flip(af.tile\
        (af.range(N_LGL), 1, N_LGL))) * lBasisArray)[:, :-1]

    nodes_tile         = af.transpose(af.tile(xi_LGL, 1, N_LGL - 1))
    power_tile         = af.flip(af.tile\
                                    (af.range(N_LGL - 1), 1, N_LGL))
    nodes_power_tile   = af.pow(nodes_tile, power_tile)

    dLp_xi = af.blas.matmul(differentiation_coeffs, nodes_power_tile)

    return dLp_xi
