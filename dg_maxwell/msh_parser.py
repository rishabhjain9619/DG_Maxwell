#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.lines as lines
import gmshtranslator.gmshtranslator as gmsh
import arrayfire as af

from dg_maxwell import isoparam
from dg_maxwell import utils
from dg_maxwell import params
from dg_maxwell import advection_2d_arbit_mesh as a2d_arbit_mesh

af.set_backend(params.backend)
af.set_device(params.device)

def read_order_2_msh(msh_file):
    '''
    Parses the :math:`2^{nd}` order **.msh** files.

    Parameters
    ----------

    msh_file : str
               **.msh** file to be parsed

    Returns
    -------

    tuple(np.ndarray, np.ndarray)
             Tuple of Nodes and Elements respectively.
             Nodes is a :math:`N \\times 2` array, where, :math:`N` is the
             total number of Nodes in the mesh. Each node contains it's
             :math:`(x, y)` coordinates.
             Elements is a :math:`N_e \\times 8` array which contains the tags
             of all the nodes which defines each element. A tag of a node is
             the array index of a node.
    '''

    msh_handler = gmsh.gmshTranslator(msh_file)

    nodes    = []
    elements = []

    def is_node(tag, x, y, z, physgroups):
        return True

    def save_node (tag, x, y, z):
        nodes.append([x, y])
        return


    def is_9_node_quadrangle (eletag, eletype,
                              physgrp, nodes):
        return eletype == msh_handler.quadrangle_9_node

    def save_element (eletag, eletype,
                      physgrp, node_tags):

        temp_nodes = node_tags.copy()
        for j, k in zip(np.arange (0,8,2), np.arange(4)):
            node_tags[j]     = temp_nodes[k]
            node_tags[j + 1] = temp_nodes[k + 4]

        # The node tag starts from 1, but now they will start from 0
        # because the nodes array indices represent the node tag.
        # Therefore (node_tags - 1) instead of (node_tags)
        elements.append(node_tags - 1)

    msh_handler.add_nodes_rule (is_node, save_node)
    msh_handler.parse()

    msh_handler.clear_rules()
    msh_handler.add_elements_rule (is_9_node_quadrangle, save_element)
    msh_handler.parse()

    nodes    = np.array(nodes)
    elements = np.array(elements, dtype = np.int)

    return nodes, elements


def plot_element_grid(x_nodes, y_nodes, xi_LGL, eta_LGL, axes_handler,
                      grid_width = 1., grid_color = 'red'):
    '''
    Uses the :math:`\\xi_{LGL}` and :math:`\\eta_{LGL}` points to plot a grid
    in the :math:`x-y` plane using the points corresponding to the
    :math:`(\\xi_{LGL}, \\eta_{LGL})` points.

    **Usage**

    .. code-block:: python
       :linenos:

       # Plots a grid for an element using 8 LGL points

       N_LGL        = 8
       xi_LGL       = lagrange.LGL_points(N)
       eta_LGL      = lagrange.LGL_points(N)

       # 8 x_nodes and y_nodes of an element
       x_nodes = [0., 0., 0., 0.5, 1., 1., 1., 0.5]
       y_nodes = [1., 0.5, 0., 0., 0., 0.5,  1., 1.]

       axes_handler = pyplot.axes()
       msh_parser.plot_element_grid(x_nodes, y_nodes,
                                    xi_LGL, eta_LGL, axes_handler)

       pyplot.title(r'Gird plot of an element.')
       pyplot.xlabel(r'$x$')
       pyplot.ylabel(r'$y$')

       pyplot.xlim(-.1, 1.1)
       pyplot.ylim(-.1, 1.1)

       pyplot.show()

    Parameters
    ----------

    x_nodes      : np.array [8]
                   x_nodes of the element.

    y_nodes      : np.array [8]
                   y_nodes of the element.

    xi_LGL       : np.array [N_LGL]
                   LGL points on the :math:`\\xi` axis

    eta_LGL      : np.array [N_LGL]
                   LGL points on the :math:`\\eta` axis

    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()

    grid_width   : float
                   Grid line width.

    grid_color   : str
                   Grid line color.

    Returns
    -------

    None
    '''

    axes_handler.set_aspect('equal')

    N = xi_LGL.shape[0]

    xy_map = np.ndarray ((N, N, 2), float)

    for m in np.arange (N):
        for n in np.arange (N):
            xy_map[m][n][0] = isoparam.isoparam_x_2D(x_nodes,
                                                     xi_LGL[m],
                                                     eta_LGL[n])
            xy_map[m][n][1] = isoparam.isoparam_y_2D(y_nodes,
                                                     xi_LGL[m],
                                                     eta_LGL[n])

    array3d = xy_map.copy()
    N = array3d.shape[0]
    #Plot the vertical lines
    for m in np.arange (0, N):
        for n in np.arange (1, N):
            line = [array3d[m][n].tolist(), array3d[m][n-1].tolist()]
            (line1_xs, line1_ys) = zip(*line)
            axes_handler.add_line(lines.Line2D(line1_xs, line1_ys,
                                               linewidth = grid_width,
                                               color = grid_color))

    #Plot the horizontal lines
    for n in np.arange (0, N):
        for m in np.arange (1, N):
            line = [array3d[m][n].tolist(), array3d[m-1][n].tolist()]
            (line1_xs, line1_ys) = zip(*line)
            axes_handler.add_line(lines.Line2D(line1_xs, line1_ys,
                                               linewidth = grid_width,
                                               color=grid_color))

    return


def plot_element_boundary(x_nodes, y_nodes, axes_handler,
                          grid_width = 2., grid_color = 'blue',
                          print_node_tag = False, node_tag_fontsize = 12):
    '''
    Plots the boundary of a given :math:`2^{nd}` order element.

    Parameters
    ----------

    x_nodes      : np.ndarray [8]
                   :math:`x` nodes of the element.

    y_nodes      : np.ndarray [8]
                   :math:`y` nodes of the element.

    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function
                   ``pyplot.axes()``.

    grid_width   : float
                   Grid line width.

    grid_color   : str
                   Grid line color.

    Returns
    -------

    None
    '''

    xi  = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)

    left_edge   = np.zeros([xi.size, 2])
    bottom_edge = np.zeros([xi.size, 2])
    right_edge  = np.zeros([xi.size, 2])
    top_edge    = np.zeros([xi.size, 2])

    left_edge[:, 0]   = isoparam.isoparam_x_2D(x_nodes, -1., eta)
    bottom_edge[:, 0] = isoparam.isoparam_x_2D(x_nodes, xi, -1)
    right_edge[:, 0]  = isoparam.isoparam_x_2D(x_nodes, 1., eta)
    top_edge[:, 0]    = isoparam.isoparam_x_2D(x_nodes, xi, 1.)

    left_edge[:, 1]   = isoparam.isoparam_y_2D(y_nodes, -1., eta)
    bottom_edge[:, 1] = isoparam.isoparam_y_2D(y_nodes, xi, -1)
    right_edge[:, 1]  = isoparam.isoparam_y_2D(y_nodes, 1., eta)
    top_edge[:, 1]    = isoparam.isoparam_y_2D(y_nodes, xi, 1.)

    # Plot edges
    utils.plot_line(left_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(bottom_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(right_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(top_edge, axes_handler, grid_width, grid_color)

    return



def plot_mesh_grid(nodes, elements, xi_LGL, eta_LGL,
                   axes_handler, print_element_tag = False,
                   element_tag_fontsize = 12,
                   print_node_tag = False, node_tag_fontsize = 12,
                   plot_LGL_grid = True):
    '''
    Plots the mesh grid.

    Parameters
    ----------

    nodes        : np.ndarray [N, 2]
                   Array of nodes in the mesh. First column and the second
                   column are the :math:`x` and :math:`y` coordinates
                   respectivily.

    elements     : np.ndarray [N_e, 8]
                   Array of elements.

    xi_LGL       : np.array [N_LGL]
                   LGL points on the :math:`\\xi` axis

    eta_LGL      : np.array [N_LGL]
                   LGL points on the :math:`\\eta` axis

    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()

    print_element_tag : bool
                        If ``True``, the function prints the elements tag at
                        the mesh plot at the centroid of each element.

    element_tag_fontsize : int
                           Fontsize of the printed element tag on the meshgrid.

    print_node_tag : bool
                     If ``True``, the function prints the node tag at
                     coordinates of the element nodes.

    node_tag_fontsize : int
                        Fontsize of the printed node tag on the meshgrid.

    plot_LGL_grid : bool
                    Plots the LGL grid if ``True``
                     
    Returns
    -------

    None
    '''

    for element_tag, element in enumerate(elements):
        if plot_LGL_grid == True:
            plot_element_grid(nodes[element, 0], nodes[element, 1],
                            xi_LGL, eta_LGL, axes_handler)
        plot_element_boundary(nodes[element, 0], nodes[element, 1],
                              axes_handler)
        if print_element_tag == True:
            element_centroid = utils.centroid(nodes[element, 0],
                                              nodes[element, 1])
            axes_handler.text(element_centroid[0], element_centroid[1],
                              str(element_tag),
                              fontsize = element_tag_fontsize, color = 'red',
                              ha = 'center',
                              va = 'center')

        if print_node_tag == True:
            for node in element[:-1]:
                axes_handler.text(nodes[node, 0], nodes[node, 1],
                                  str(node),
                                  fontsize = node_tag_fontsize)

    return


def edge_location(node_indices):
    '''
    Finds the edge id to which given nodes, of a :math:`2^{nd}` order
    quadrangular mesh, belong.
    
    +-------------+-------------+
    | **Edge**    | **Edge ID** |
    +-------------+-------------+
    | Left Edge   | :math:`0`   |
    +-------------+-------------+
    | Bottom Edge | :math:`1`   |
    +-------------+-------------+
    | Right Edge  | :math:`2`   |
    +-------------+-------------+
    | Top Edge    | :math:`3`   |
    +-------------+-------------+

    Parameters
    ----------
    node_indices : np.array [3]
                   Node id of the nodes belong to an edge.

    Returns
    -------
        int
        The edge id corresponding to an edge. If it is not an edge,
        then returns ``None``
    '''
    
    left_edge = np.array([0, 1, 2])
    bottom_edge = np.array([2, 3, 4])
    right_edge = np.array([4, 5, 6])
    top_edge = np.array([0, 6, 7])

    if len(np.intersect1d(left_edge, node_indices)) == 3:
        return 0

    if len(np.intersect1d(bottom_edge, node_indices)) == 3:
        return 1

    if len(np.intersect1d(right_edge, node_indices)) == 3:
        return 2

    if len(np.intersect1d(top_edge, node_indices)) == 3:
        return 3
    
    return None


def interelement_relations(elements):
    '''
    Finds the neighbouring elements and the physical boundaries for each of
    the elements.

    Parameters
    ----------
    elements : np.array [N_elements 9]
               The elements of a :math:`2^{nd}` order quadrangular meshgrid.

    Returns
    -------
    element_relations : np.array[N_elements 4]
                        Each element tag has :math:`4` indices, one for each
                        edge. If the element has an edge common with some
                        element, then that edge id stores the tag of the
                        element with which it has common edge. If an edge of
                        an element is a physical boundary, then it's value is
                        :math:`-1`. Example, if element :math:`0` has its
                        bottom edge common with element :math:`4` and its right
                        edge common with element :math:`1`, the for this
                        element, then the elements relations array for this
                        element will will be ``[-1, 4, 1, -1]``.
                        
                        To find what each edge id represents, see this
                        :py:meth:`dg_maxwell.msh_parser.edge_location`.
    '''
    element_relations = np.ones([elements.shape[0], 4], dtype = np.int32) * -1


    for element_0_tag in np.arange(elements.shape[0]):
        element_0 = elements[element_0_tag]

        for element_tag in np.delete(np.arange(elements.shape[0]),
                                     element_0_tag, axis = 0):
            common_node_indices = np.nonzero(np.in1d(element_0,
                                                     elements[element_tag]))[0]
            if len(common_node_indices) == 3:
                element_relations[
                    element_0_tag,
                    edge_location(common_node_indices)] = element_tag
    
    return element_relations



def identify_element_physical_edges(elements, advec_var):
    '''
    For an element it tdentifies the left, bottom, right,
    and top edges physically. It assigns each edge of an
    element an id depending on the table shown below.
    +-------------+----------------------+
    | **Edge**    | **Assigned Edge ID** |
    +-------------+----------------------+
    | Left Edge   | :math:`0`            |
    +-------------+----------------------+
    | Bottom Edge | :math:`1`            |
    +-------------+----------------------+
    | Right Edge  | :math:`2`            |
    +-------------+----------------------+
    | Top Edge    | :math:`3`            |
    +-------------+----------------------+
    
    Parameters
    ----------
    elements : np.array([N_elements 9], dtype = np.int)
               The element array returned by
               :py:meth:`dg_maxwell.msh_parser.read_order_2_msh`.

    advec_var : :py:meth:`dg_maxwell.global_variables.advection_variables`

    Returns
    -------
    physical_edge_identity : af.Array([N_elements 4 1 1], dtype = af.Dtype.int64)
                             The physical edge id for each edge of each elements.
                             See table above.
    '''
    x_edge = af.constant(0., d0 = advec_var.elements.shape[0],
                         d1 = params.N_LGL, d2 = 4, dtype = af.Dtype.f64)
    y_edge = af.constant(0., d0 = advec_var.elements.shape[0],
                         d1 = params.N_LGL, d2 = 4, dtype = af.Dtype.f64)

    for edge_id in np.arange(4):
        x_edge[:, :, edge_id] = af.transpose(
            a2d_arbit_mesh.u_at_edge(advec_var.x_e_ij,
                                     edge_id, advec_var))
        y_edge[:, :, edge_id] = af.transpose(
            a2d_arbit_mesh.u_at_edge(advec_var.y_e_ij,
                                     edge_id, advec_var))


    ###################################################################################

    x_edge_c = af.mean(x_edge, dim = 1)
    y_edge_c = af.mean(y_edge, dim = 1)

    x_quad_c = af.mean(af.mean(x_edge, dim = 1), dim = 2)
    y_quad_c = af.mean(af.mean(y_edge, dim = 1), dim = 2)

    ###################################################################################

    delta_x_edge_c = af.reorder(af.broadcast(utils.add, x_edge_c, -x_quad_c),
                                d0 = 0, d1 = 2, d2 = 1)
    delta_y_edge_c = af.reorder(af.broadcast(utils.add, y_edge_c, -y_quad_c),
                                d0 = 0, d1 = 2, d2 = 1)

    ###################################################################################

    delta_xy_complex = np.array(delta_x_edge_c + 1j * delta_y_edge_c)

    ###################################################################################

    edge_angle = af.np_to_af_array(np.angle(delta_xy_complex, deg = True))
    angle_less_than_zero = af.cast(edge_angle < 0, dtype = af.Dtype.f64)
    edge_angle = edge_angle + angle_less_than_zero * 360

    ###################################################################################

    # Identifying physical right edges
    physical_right_edge = (edge_angle >= 315.) * (edge_angle <= 360) + (edge_angle >= 0.) * (edge_angle <= 45)

    # Identifying physical top edges
    physical_top_edge = (edge_angle >= 45) * (edge_angle <= 135)

    # Identifying physical left edges
    physical_left_edge = (edge_angle >= 135) * (edge_angle <= 225)

    # Identifying physical left edges
    physical_bottom_edge = (edge_angle >= 225) * (edge_angle < 315)

    ###################################################################################

    physical_edge_identity = physical_left_edge   * 1 \
                           + physical_bottom_edge * 2 \
                           + physical_right_edge  * 3 \
                           + physical_top_edge    * 4 \
                           - 1

    return physical_edge_identity



def get_edge_nodes(elements, element_tag, edge_id):
    '''
    Gives the edge nodes of an element for an edge id.
    To find what each edge id represents, see this
    :py:meth:`dg_maxwell.msh_parser.edge_location`.
    
    Parameters
    ----------
    elements : np.array([N_elements 9], dtype = np.int)
               The element array returned by
               :py:meth:`dg_maxwell.msh_parser.read_order_2_msh`.

    element_tag : unsigned int
                  The element tag for which an edge has to be returned.

    edge_id : unsigned int
              The edge to look for in an element

    Return
    ------
    edge_nodes_tag : np.Array([3], dtype= np.int64)
                     The node tag which makes up an edge.
    '''
    edge_nodes_tag = None
    
    if edge_id < 3:
        edge_nodes_tag =  elements[element_tag, edge_id * 2:(edge_id * 2 + 3)]
    else:
        edge_nodes_tag = np.append(elements[element_tag, edge_id * 2:(edge_id * 2 + 2)],
                                   elements[element_tag, 0])
    
    return edge_nodes_tag


def edge_nodes_reordered(elements, element_tag, edge_id,
                         return_as_edge, advec_var):
    '''
    Finds the edge nodes of an element and reorders the nodes of an edge
    according to the requested ``return_as_edge`` variable.
    
    Parameters
    ----------
    elements : np.array([N_elements 9], dtype = np.int)
               The element array returned by
               :py:meth:`dg_maxwell.msh_parser.read_order_2_msh`.

    element_tag : unsigned int
                  The element tag for which an edge has to be returned.

    edge_id : unsigned int
              The edge to look for in an element

    return_as_edge : unsigned int
                     The edge as which the ``edge_id`` edge
                     has to be returned.

    advec_var : :py:meth:`dg_maxwell.global_variables.advection_variables`

    Returns
    -------
    edge_nodes_tag : np.Array([3], dtype= np.int64)
                     The edge with the rearranged node order.
    '''

    edge_nodes_tag = get_edge_nodes(elements, element_tag, edge_id)

    # Left edge case
    if return_as_edge == 0:

        # Rearrange the left edge nodes
        element_tag_left_edge = edge_nodes_tag

        # Arrange the nodes for left nodes
        y_nodes = advec_var.nodes[element_tag_left_edge][:, 1]

        y_nodes_dic_node_tag = np.zeros([y_nodes.shape[0], 2])
        y_nodes_dic_node_tag[:, 0] = y_nodes
        y_nodes_dic_node_tag[:, 1] = element_tag_left_edge

        y_nodes_argsort = np.argsort(y_nodes_dic_node_tag, axis = 0)
        left_edge_nodes = np.flipud(element_tag_left_edge[y_nodes_argsort[:, 0]])

        edge_nodes_tag = left_edge_nodes

    # Bottom edge case
    if return_as_edge == 1:
        # Rearrange the bottom edge nodes.
        element_tag_bottom_edge = edge_nodes_tag

        # Arrange the nodes for bottom nodes
        x_nodes = advec_var.nodes[element_tag_bottom_edge][:, 0]

        x_nodes_dic_node_tag = np.zeros([x_nodes.shape[0], 2])
        x_nodes_dic_node_tag[:, 0] = x_nodes
        x_nodes_dic_node_tag[:, 1] = element_tag_bottom_edge

        x_nodes_argsort = np.argsort(x_nodes_dic_node_tag, axis = 0)
        bottom_edge_nodes = element_tag_bottom_edge[x_nodes_argsort[:, 0]]
        
        edge_nodes_tag = bottom_edge_nodes

    # Right edge case
    if return_as_edge == 2:
        # Rearrange the right edge nodes
        element_tag_right_edge = edge_nodes_tag

        # Arrange the nodes for right nodes
        y_nodes = advec_var.nodes[element_tag_right_edge][:, 1]

        y_nodes_dic_node_tag = np.zeros([y_nodes.shape[0], 2])
        y_nodes_dic_node_tag[:, 0] = y_nodes
        y_nodes_dic_node_tag[:, 1] = element_tag_right_edge

        y_nodes_argsort = np.argsort(y_nodes_dic_node_tag, axis = 0)
        right_edge_nodes = element_tag_right_edge[y_nodes_argsort[:, 0]]

        edge_nodes_tag = right_edge_nodes

    # Top edge case
    if return_as_edge == 3:
        # Rearrange the right edge nodes
        element_tag_top_edge = edge_nodes_tag

        # Arrange the nodes for top nodes
        x_nodes = advec_var.nodes[element_tag_top_edge][:, 0]

        x_nodes_dic_node_tag = np.zeros([x_nodes.shape[0], 2])
        x_nodes_dic_node_tag[:, 0] = x_nodes
        x_nodes_dic_node_tag[:, 1] = element_tag_top_edge

        x_nodes_argsort = np.argsort(x_nodes_dic_node_tag, axis = 0)
        top_edge_nodes = np.flipud(element_tag_top_edge[x_nodes_argsort[:, 0]])
        
        edge_nodes_tag = top_edge_nodes

    return edge_nodes_tag


def rearrange_element_edges(elements, advec_var):
    '''
    In a mesh read from mesh parser none of the edge have the sense of the
    their physical identity, i.e., they don't know whether they are left,
    bottom, right and top edge. This function reads the original mesh and
    rearranges the nodes of an element such that the first edge is always
    the physically located left edge, second edge is the physically located
    bottom edge, third edge is the physically located right edge, and the
    fourth edge is the physically located top edge.
    
    Parameters
    ----------
    elements : np.array([N_elements 9], dtype = np.int)
               The element array returned by
               :py:meth:`dg_maxwell.msh_parser.read_order_2_msh`.

    advec_var : :py:meth:`dg_maxwell.global_variables.advection_variables`

    Returns
    -------
    new_element_edge_ordering : np.Array(advec_var.elements.shape, dtype = np.int64)
                                Elements with reordered edges.
    '''
    new_element_edge_ordering = np.zeros(advec_var.elements.shape, dtype = np.int64)
    edge_reorder = identify_element_physical_edges(elements, advec_var)

    for element_tag in np.arange(advec_var.elements.shape[0]):
        for edge_id in np.arange(4):
            return_as_edge_id = int(af.sum(edge_reorder[element_tag, edge_id]))
            new_element_edge_ordering[
                element_tag,
                return_as_edge_id * 2:(return_as_edge_id * 2 + 3)] = \
                    edge_nodes_reordered(elements, element_tag, edge_id,
                                         af.sum(edge_reorder[element_tag, edge_id]),
                                         advec_var = advec_var)
    
    return new_element_edge_ordering
