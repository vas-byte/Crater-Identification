import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
sys.path.append("../crater_pose_estimation-main/")

from src.utils import *

def matrix_adjugate(matrix):
    # Helper function to compute the determinant of a 2x2 matrix
    def det_2x2(m):
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    # Compute the cofactor matrix
    cofactor_matrix = [
        [
            (-1)**(i+j) * det_2x2([
                [matrix[(i+1)%3][(j+1)%3], matrix[(i+1)%3][(j+2)%3]],
                [matrix[(i+2)%3][(j+1)%3], matrix[(i+2)%3][(j+2)%3]]
            ])
            for j in range(3)
        ]
        for i in range(3)
    ]

    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(3)] for i in range(3)]

    return np.array(adjugate_matrix)

# def coplanar_triad_descriptors(Ai, Aj, Ak):
#     """Generate the seven unique invariants for a triad of coplanar conics.
#
#     Equations (128) through (131) in Christian et al. (2021)
#
#     Args:
#         Ai (ArrayLike): Matrix representation of the first conic with dims (...,3,3)
#         Aj (ArrayLike): Matrix representation of the second conic
#         Ak (ArrayLike): Matrix representation of the third conic
#
#     Returns:
#         np.ndarray: The seven invariants with dims (...,7)
#     """
#
#     # Normalise matrices such that the determinants are all 1
#     Ai = np.cbrt(1 / np.linalg.det(Ai))[..., np.newaxis, np.newaxis] * Ai
#     Aj = np.cbrt(1 / np.linalg.det(Aj))[..., np.newaxis, np.newaxis] * Aj
#     Ak = np.cbrt(1 / np.linalg.det(Ak))[..., np.newaxis, np.newaxis] * Ak
#
#     Aiinv = np.linalg.inv(Ai)
#     Ajinv = np.linalg.inv(Aj)
#     Akinv = np.linalg.inv(Ak)
#
#     Iij = np.trace(Aiinv @ Aj, axis1=-2, axis2=-1)
#     Iji = np.trace(Ajinv @ Ai, axis1=-2, axis2=-1)
#     Iik = np.trace(Aiinv @ Ak, axis1=-2, axis2=-1)
#     Iki = np.trace(Akinv @ Ai, axis1=-2, axis2=-1)
#     Ijk = np.trace(Ajinv @ Ak, axis1=-2, axis2=-1)
#     Ikj = np.trace(Akinv @ Aj, axis1=-2, axis2=-1)
#     Iijk = np.trace(
#         (matrix_adjugate(Aj + Ak) - matrix_adjugate(Aj - Ak)) @ Ai, axis1=-2, axis2=-1
#     )
#
#     return np.stack([Iij, Iji, Iik, Iki, Ijk, Ikj, Iijk], axis=-1)


def cofactor(matrix):
    """
    Compute the cofactor matrix of a 3x3 matrix.

    Args:
    - matrix (np.array): A 3x3 numpy array.

    Returns:
    - np.array: The cofactor matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # Initialize the cofactor matrix
    C = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            # Create the submatrix by removing the i-th row and j-th column
            submatrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)

            # Compute the determinant of the 2x2 submatrix
            det = np.linalg.det(submatrix)

            # Compute the cofactor
            C[i, j] = (-1) ** (i + j) * det

    return C

def non_coplanar_triad_descriptors(A1, A2, A3):
    """Algorithm 1 on page 1104 in Christian et al. (2021)"""

    # TODO vectorise
    ls = []
    for Ai, Aj in ((A1, A2), (A2, A3), (A3, A1)):
        eigs, _ = np.linalg.eig(Aj @ np.linalg.inv(-Ai))
        Bij, z_bar = None, None
        for eig_index in range(3):
            eig = eigs[..., eig_index]
            Bij = eig * Ai + Aj
            # Bij_star = matrix_adjugate(Bij)
            Bij_star = cofactor(Bij)
            Bij_star = Bij_star.T
            k = np.argmax(np.abs(np.diagonal(Bij_star)), axis=-1)
            z_bar = -Bij_star[..., k] / np.emath.sqrt(-Bij_star[..., k, k])
            if np.isreal(z_bar).all() and not (np.isnan(z_bar).any()):
                break
        # z_bar_cross = np.dot(np.atleast_2d(z_bar).T, np.atleast_2d(z_bar))  # ?? TODO
        z_bar_cross = np.array(
            [
                [0, -z_bar[2], z_bar[1]],
                [z_bar[2], 0, -z_bar[0]],
                [-z_bar[1], z_bar[0], 0],
            ]
        )
        D = Bij + z_bar_cross
        k, l = np.unravel_index(np.argmax(np.abs(D)), D.shape)
        h = np.array([[D[k, 0], D[k, 1], D[k, 2]]]).T
        g = np.array([[D[0, l], D[1, l], D[2, l]]]).T

        _, Ai_x, Ai_y, Ai_a, Ai_b, Ai_phi = extract_ellipse_parameters_from_conic(Ai)
        _, Aj_x, Aj_y, Aj_a, Aj_b, Aj_phi = extract_ellipse_parameters_from_conic(Aj)

        i_centre = np.array([[Ai_x, Ai_y, 1]]).T
        j_centre = np.array([[Aj_x, Aj_y, 1]]).T
        lij = (
            g if np.sign(np.dot(g.T, i_centre)) != np.sign(np.dot(g.T, j_centre)) else h
        )
        ls.append(lij)

    Aistar = cofactor(A1)
    Ajstar = cofactor(A2)
    Akstar = cofactor(A3)

    Aistar = Aistar.T
    Ajstar = Ajstar.T
    Akstar = Akstar.T

    lij, ljk, lik = ls  # all column vectors

    J1 = np.arccosh(
        np.linalg.norm(lij.T @ Aistar @ lik)
        / np.sqrt((lij.T @ Aistar @ lij) @ (lik.T @ Aistar @ lik))
    )[0, 0]
    J2 = np.arccosh(
        np.linalg.norm(lij.T @ Ajstar @ ljk)
        / np.sqrt((lij.T @ Ajstar @ lij) @ (ljk.T @ Ajstar @ ljk))
    )[0, 0]
    J3 = np.arccosh(
        np.linalg.norm(lik.T @ Akstar @ ljk)
        / np.sqrt((lik.T @ Akstar @ lik) @ (ljk.T @ Akstar @ ljk))
    )[0, 0]
    if not np.isreal([J1, J2, J3]).all():
        raise ValueError("Complex descriptors generated - craters should not intersect")
    return np.real([J1, J2, J3])
#
#
# def non_coplanar_triad_descriptors_vec(A1, A2, A3):
#     """Algorithm 1 on page 1104 in Christian et al. (2021)"""
#     ls = []
#     pairs = np.stack(((A1, A2), (A2, A3), (A3, A1)), axis=0)
#     eigs = np.linalg.eig(pairs[:, 1] @ np.linalg.inv(-pairs[:, 0]))[0]
#
#     # This part is tricky to vectorize due to the conditional break.
#     # It might be more efficient to keep it as a loop.
#     for i in range(3):
#         for eig_index in range(3):
#             eig = eigs[i, eig_index]
#             Ai = pairs[i, 0]
#             Aj = pairs[i, 1]
#             Bij = eig * Ai + Aj
#             Bij_star = cofactor(Bij).T
#             k = np.argmax(np.abs(np.diagonal(Bij_star)))
#             z_bar = -Bij_star[..., k] / np.emath.sqrt(-Bij_star[..., k, k])
#             if np.isreal(z_bar).all() and not (np.isnan(z_bar).any()):
#                 break
#
#         z_bar_cross = np.array(
#             [
#                 [0, -z_bar[2], z_bar[1]],
#                 [z_bar[2], 0, -z_bar[0]],
#                 [-z_bar[1], z_bar[0], 0],
#             ]
#         )
#         D = Bij + z_bar_cross
#         k, l = np.unravel_index(np.argmax(np.abs(D)), D.shape)
#         h = np.array([[D[k, 0], D[k, 1], D[k, 2]]]).T
#         g = np.array([[D[0, l], D[1, l], D[2, l]]]).T
#
#         _, Ai_x, Ai_y, Ai_a, Ai_b, Ai_phi = extract_ellipse_parameters_from_conic(Ai)
#         _, Aj_x, Aj_y, Aj_a, Aj_b, Aj_phi = extract_ellipse_parameters_from_conic(Aj)
#
#         i_centre = np.array([[Ai_x, Ai_y, 1]]).T
#         j_centre = np.array([[Aj_x, Aj_y, 1]]).T
#         lij = (
#             g if np.sign(np.dot(g.T, i_centre)) != np.sign(np.dot(g.T, j_centre)) else h
#         )
#         ls.append(lij)
#
#     Aistar = cofactor(A1)
#     Ajstar = cofactor(A2)
#     Akstar = cofactor(A3)
#
#     Aistar = Aistar.T
#     Ajstar = Ajstar.T
#     Akstar = Akstar.T
#
#     lij, ljk, lik = ls  # all column vectors
#
#     J1 = np.arccosh(
#         np.linalg.norm(lij.T @ Aistar @ lik)
#         / np.sqrt((lij.T @ Aistar @ lij) @ (lik.T @ Aistar @ lik))
#     )[0, 0]
#     J2 = np.arccosh(
#         np.linalg.norm(lij.T @ Ajstar @ ljk)
#         / np.sqrt((lij.T @ Ajstar @ lij) @ (ljk.T @ Ajstar @ ljk))
#     )[0, 0]
#     J3 = np.arccosh(
#         np.linalg.norm(lik.T @ Akstar @ ljk)
#         / np.sqrt((lik.T @ Akstar @ lik) @ (ljk.T @ Akstar @ ljk))
#     )[0, 0]
#     if not np.isreal([J1, J2, J3]).all():
#         raise ValueError("Complex descriptors generated - craters should not intersect")
#
#     return np.real([J1, J2, J3])


def non_coplanar_triad_descriptors_w_flag(A1, A2, A3):
    """Algorithm 1 on page 1104 in Christian et al. (2021)"""

    # test adjugate matrix
    # adj_A1 = matrix_adjugate(A1)
    # adj_A1 = cofactor(A1)
    # detA_Aiinv = np.linalg.det(A1) * np.linalg.inv(A1)
    # print(np.linalg.norm(adj_A1 - detA_Aiinv))
    #
    # testing_matrix = np.array([[-3, 2, -5], [-1,0, -2], [3, -4, 1]])
    # test_mat_adj = cofactor(testing_matrix)
    # test_mat_adj = test_mat_adj.T
    # test_det_inv = np.linalg.det(testing_matrix) * np.linalg.inv(testing_matrix)
    # print(np.linalg.norm(test_mat_adj - test_det_inv))
    #
    # # testing_matrix = np.array([[-3, 2, -5], [-1, 0, -2], [3, -4, 1]])
    # # test_mat_adj = matrix_adjugate(testing_matrix)
    # # test_det_inv = np.linalg.det(testing_matrix) * np.linalg.inv(testing_matrix)
    # # print(np.linalg.norm(test_mat_adj - test_det_inv))
    #
    #
    # adj_Aj = matrix_adjugate(A2)
    # detA_Ajinv = np.linalg.det(A2) * np.linalg.inv(A2)
    # print(np.linalg.norm(adj_Aj - detA_Ajinv))
    #
    # adj_Ak = matrix_adjugate(A3)
    # detA_Akinv = np.linalg.det(A3) * np.linalg.inv(A3)
    # print(np.linalg.norm(adj_Ak - detA_Akinv))
    #

    # TODO vectorise
    ls = []
    for Ai, Aj in ((A1, A2), (A2, A3), (A3, A1)):
        eigs, _ = np.linalg.eig(Aj @ np.linalg.inv(-Ai))
        Bij, z_bar = None, None
        for eig_index in range(3):
            eig = eigs[..., eig_index]
            Bij = eig * Ai + Aj
            # Bij_star = matrix_adjugate(Bij)
            Bij_star = cofactor(Bij)
            Bij_star = Bij_star.T
            k = np.argmax(np.abs(np.diagonal(Bij_star)), axis=-1)
            z_bar = -Bij_star[..., k] / np.emath.sqrt(-Bij_star[..., k, k])
            if np.isreal(z_bar).all() and not (np.isnan(z_bar).any()):
                break
        # z_bar_cross = np.dot(np.atleast_2d(z_bar).T, np.atleast_2d(z_bar))  # ?? TODO
        z_bar_cross = np.array(
            [
                [0, -z_bar[2], z_bar[1]],
                [z_bar[2], 0, -z_bar[0]],
                [-z_bar[1], z_bar[0], 0],
            ]
        )
        D = Bij + z_bar_cross
        k, l = np.unravel_index(np.argmax(np.abs(D)), D.shape)
        h = np.array([[D[k, 0], D[k, 1], D[k, 2]]]).T
        g = np.array([[D[0, l], D[1, l], D[2, l]]]).T

        _, Ai_x, Ai_y, Ai_a, Ai_b, Ai_phi = extract_ellipse_parameters_from_conic(Ai)
        _, Aj_x, Aj_y, Aj_a, Aj_b, Aj_phi = extract_ellipse_parameters_from_conic(Aj)

        i_centre = np.array([[Ai_x, Ai_y, 1]]).T
        j_centre = np.array([[Aj_x, Aj_y, 1]]).T
        lij = (
            g if np.sign(np.dot(g.T, i_centre)) != np.sign(np.dot(g.T, j_centre)) else h
        )
        ls.append(lij)

    Aistar = cofactor(A1)
    Ajstar = cofactor(A2)
    Akstar = cofactor(A3)

    Aistar = Aistar.T
    Ajstar = Ajstar.T
    Akstar = Akstar.T

    lij, ljk, lik = ls  # all column vectors

    J1 = np.arccosh(
        np.linalg.norm(lij.T @ Aistar @ lik)
        / np.sqrt((lij.T @ Aistar @ lij) @ (lik.T @ Aistar @ lik))
    )[0, 0]
    J2 = np.arccosh(
        np.linalg.norm(lij.T @ Ajstar @ ljk)
        / np.sqrt((lij.T @ Ajstar @ lij) @ (ljk.T @ Ajstar @ ljk))
    )[0, 0]
    J3 = np.arccosh(
        np.linalg.norm(lik.T @ Akstar @ ljk)
        / np.sqrt((lik.T @ Akstar @ lik) @ (ljk.T @ Akstar @ ljk))
    )[0, 0]
    if not np.isreal([J1, J2, J3]).all():
        return True, np.array([0, 0, 0])
        # raise ValueError("Complex descriptors generated - craters should not intersect")
    return False, np.real([J1, J2, J3])


def plot_ellipses(ellipses, line):
    """
    Plot multiple ellipses using matplotlib's Ellipse patch and highlight pixels on a given line.

    Parameters:
    - ellipses: List of dictionaries, each representing an ellipse.
                Each dictionary should have keys: 'a', 'b', 'h', 'k', 'angle', and 'color'.
    - line: Tuple representing the line in homogeneous coordinates (a, b, c).
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Variables to store the min and max values for x and y to set the axis limits
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for ellipse_params in ellipses:
        a = ellipse_params['a']
        b = ellipse_params['b']
        h = ellipse_params.get('h', 0)
        k = ellipse_params.get('k', 0)
        angle = ellipse_params.get('angle', 0)
        color = ellipse_params.get('color', 'blue')

        # Update the axis limits
        x_min = min(x_min, h - a)
        x_max = max(x_max, h + a)
        y_min = min(y_min, k - b)
        y_max = max(y_max, k + b)

        # Create an Ellipse patch
        ellipse = patches.Ellipse((h, k), 2 * a, 2 * b, angle=angle, color=color, alpha=0.5)
        ax.add_patch(ellipse)

    # Set the axis limits
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    # Check each pixel if it lies on the given line
    a, b, c = line
    y_vals, x_vals = np.mgrid[y_min - 1:y_max + 1:0.1, x_min - 1:x_max + 1:0.1]
    mask = np.abs(a * x_vals + b * y_vals + c) < 1  # Threshold to determine if a pixel lies on the line

    ax.imshow(mask, extent=[x_min - 1, x_max + 1, y_min - 1, y_max + 1], origin='lower', cmap='gray_r', alpha=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    plt.title("Multiple Ellipse Visualization with Line")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend([patches.Patch(color=ellipse['color']) for ellipse in ellipses],
               [f"Ellipse {i + 1}" for i in range(len(ellipses))])
    plt.show()


def non_coplanar_triad_descriptors_investigation(A1, A2, A3, P1, P2, P3):
    """Algorithm 1 on page 1104 in Christian et al. (2021)"""
    # TODO vectorise
    ls = []
    for Ai, Aj, Pi, Pj in ((A1, A2, P1, P2), (A2, A3, P2, P3), (A3, A1, P3, P1)):
        eigs, _ = np.linalg.eig(Aj @ np.linalg.inv(-Ai))
        Bij, z_bar = None, None
        for eig_index in range(3):
            eig = eigs[..., eig_index]
            Bij = eig * Ai + Aj
            # Bij_star = matrix_adjugate(Bij)
            Bij_star = cofactor(Bij)
            Bij_star = Bij_star.T
            k = np.argmax(np.abs(np.diagonal(Bij_star)), axis=-1)
            z_bar = -Bij_star[..., k] / np.emath.sqrt(-Bij_star[..., k, k])
            if np.isreal(z_bar).all() and not (np.isnan(z_bar).any()):
                break
        # z_bar_cross = np.dot(np.atleast_2d(z_bar).T, np.atleast_2d(z_bar))  # ?? TODO
        z_bar_cross = np.array(
            [
                [0, -z_bar[2], z_bar[1]],
                [z_bar[2], 0, -z_bar[0]],
                [-z_bar[1], z_bar[0], 0],
            ]
        )
        D = Bij + z_bar_cross
        k, l = np.unravel_index(np.argmax(np.abs(D)), D.shape)
        h = np.array([[D[k, 0], D[k, 1], D[k, 2]]]).T
        g = np.array([[D[0, l], D[1, l], D[2, l]]]).T

        _, Ai_x, Ai_y, Ai_a, Ai_b, Ai_phi = extract_ellipse_parameters_from_conic(Ai)
        _, Aj_x, Aj_y, Aj_a, Aj_b, Aj_phi = extract_ellipse_parameters_from_conic(Aj)

        i_centre = np.array([[Ai_x, Ai_y, 1]]).T
        j_centre = np.array([[Aj_x, Aj_y, 1]]).T

        ## the line is represented in the homogeneous coordinate, i.e., ax + by + cw = 0.
        # dot product checks the SIGNED distance between a point and the line, If both of them have the opposite signs,
        # means that they fall on different side of the line
        lij = (
            g if np.sign(np.dot(g.T, i_centre)) != np.sign(np.dot(g.T, j_centre)) else h
        )
        ls.append(lij)


    Aistar = cofactor(A1)
    Ajstar = cofactor(A2)
    Akstar = cofactor(A3)

    Aistar = Aistar.T
    Ajstar = Ajstar.T
    Akstar = Akstar.T

    lij, ljk, lik = ls  # all column vectors

    J1 = np.arccosh(
        np.linalg.norm(lij.T @ Aistar @ lik)
        / np.sqrt((lij.T @ Aistar @ lij) @ (lik.T @ Aistar @ lik))
    )[0, 0]
    J2 = np.arccosh(
        np.linalg.norm(lij.T @ Ajstar @ ljk)
        / np.sqrt((lij.T @ Ajstar @ lij) @ (ljk.T @ Ajstar @ ljk))
    )[0, 0]
    J3 = np.arccosh(
        np.linalg.norm(lik.T @ Akstar @ ljk)
        / np.sqrt((lik.T @ Akstar @ lik) @ (ljk.T @ Akstar @ ljk))
    )[0, 0]
    if not np.isreal([J1, J2, J3]).all():
        raise ValueError("Complex descriptors generated - craters should not intersect")
    return np.real([J1, J2, J3])

def coplanar_pair_descriptors(Ai, Aj):
    Ai = np.cbrt(1 / np.linalg.det(Ai))[..., np.newaxis, np.newaxis] * Ai
    Aj = np.cbrt(1 / np.linalg.det(Aj))[..., np.newaxis, np.newaxis] * Aj
    
    Aiinv = np.linalg.inv(Ai)
    Ajinv = np.linalg.inv(Aj)

    Iij = np.trace(Aiinv @ Aj)
    Iji = np.trace(Ajinv @ Ai)

    return np.stack([Iij, Iji], axis=-1)

def triad_triangle_descriptors(id_1, id_2, id_3, CW_1, CW_2, CW_3):
    # first sort them based on interior angles
    r_12 = CW_2[0:3] - CW_1[0:3]
    r_13 = CW_3[0:3] - CW_1[0:3]
    r_23 = CW_3[0:3] - CW_2[0:3]
    
    plane_center = (CW_1[0:3] + CW_2[0:3] + CW_3[0:3]) / 3
    plane_center = plane_center / np.linalg.norm(plane_center)
    
    r_combs = {'r_12': r_12, 'r_21': -r_12, 'r_13': r_13, 'r_31': -r_13, 'r_23': r_23, 'r_32': -r_23}
    cw_combs = {'c_1': CW_1, 'c_2': CW_2, 'c_3': CW_3}
    id_combs = {'id_1': id_1, 'id_2': id_2, 'id_3': id_3}
    
    r_12_norm = r_12 / np.linalg.norm(r_12)
    r_13_norm = r_13 / np.linalg.norm(r_13)
    r_23_norm = r_23 / np.linalg.norm(r_23)
    
    # Calculate the angles using the dot product
    angle1 = np.arccos((np.dot(r_12_norm, r_13_norm)))  # Angle between r_12 and r_13
    angle2 = np.arccos((np.dot(-r_12_norm, r_23_norm))) # Angle between -r_12 and r_23
    angle3 = np.arccos((np.dot(r_13_norm, r_23_norm)))  # Angle between r_13 and r_23
    
    # Use argsort to get sorted indices
    # Store the angles in an array
    angles = np.array([angle1, angle2, angle3])
    
    sorted_indices = np.argsort([angle1, angle2, angle3])
    
    alpha_s = np.cos(angles[sorted_indices[0]])
    alpha_l = np.cos(angles[sorted_indices[2]])
    
    sorted_indices = sorted_indices + 1
    r_ij = r_combs['r_' + str(sorted_indices[0]) + str(sorted_indices[1])]
    r_jk = r_combs['r_' + str(sorted_indices[1]) + str(sorted_indices[2])]
    r_ki = r_combs['r_' + str(sorted_indices[2]) + str(sorted_indices[1])]
    
    di = cw_combs['c_' + str(sorted_indices[0])][3]
    dj = cw_combs['c_' + str(sorted_indices[1])][3]
    dk = cw_combs['c_' + str(sorted_indices[2])][3]
    
    gamma_s = di / np.linalg.norm(r_ij)
    gamma_m = dj / np.linalg.norm(r_ij)
    gamma_l = dk / np.linalg.norm(r_ij)
    
    orientation = determine_orientation(r_ij, r_jk, r_ki, plane_center)
    
    id_i = id_combs['id_' + str(sorted_indices[0])]
    id_j = id_combs['id_' + str(sorted_indices[1])]
    id_k = id_combs['id_' + str(sorted_indices[2])]
    
    return np.array([id_i, id_j, id_k, alpha_s, alpha_l, gamma_s, gamma_m, gamma_l]), orientation

def determine_order_on_plane(p1, p2, p3):
    """
    Determines whether three points are in clockwise or counter-clockwise order.

    Args:
    p1, p2, p3: Tuples representing the pixel coordinates (x, y) of the points

    Returns:
    A string indicating the order ('Clockwise', 'Counter-Clockwise', or 'Collinear')
    """

    # Extracting coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculating the cross product of vectors (p1p2) and (p1p3)
    z = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    # Determining the order based on the sign of Z
    if z > 0:
        return -1
    elif z < 0:
        # return "Clockwise"
        return 1
    
def triad_triangle_descriptors_on_plane(id_1, id_2, id_3, CW_1, CW_2, CW_3):
    # first sort them based on interior angles
    r_12 = CW_2[0:2] - CW_1[0:2]
    r_13 = CW_3[0:2] - CW_1[0:2]
    r_23 = CW_3[0:2] - CW_2[0:2]
    
    r_combs = {'r_12': r_12, 'r_21': -r_12, 'r_13': r_13, 'r_31': -r_13, 'r_23': r_23, 'r_32': -r_23}
    cw_combs = {'c_1': CW_1, 'c_2': CW_2, 'c_3': CW_3}
    id_combs = {'id_1': id_1, 'id_2': id_2, 'id_3': id_3}
    
    r_12_norm = r_12 / np.linalg.norm(r_12)
    r_13_norm = r_13 / np.linalg.norm(r_13)
    r_23_norm = r_23 / np.linalg.norm(r_23)
    
    # Calculate the angles using the dot product
    angle1 = np.arccos((np.dot(r_12_norm, r_13_norm)))  # Angle between r_12 and r_13
    angle2 = np.arccos((np.dot(-r_12_norm, r_23_norm))) # Angle between -r_12 and r_23
    angle3 = np.arccos((np.dot(r_13_norm, r_23_norm)))  # Angle between r_13 and r_23
    
    # Use argsort to get sorted indices
    # Store the angles in an array
    angles = np.array([angle1, angle2, angle3])
    
    sorted_indices = np.argsort([angle1, angle2, angle3])
    
    alpha_s = np.cos(angles[sorted_indices[0]])
    alpha_l = np.cos(angles[sorted_indices[2]])
    
    sorted_indices = sorted_indices + 1
    r_ij = r_combs['r_' + str(sorted_indices[0]) + str(sorted_indices[1])]
    r_jk = r_combs['r_' + str(sorted_indices[1]) + str(sorted_indices[2])]
    r_ki = r_combs['r_' + str(sorted_indices[2]) + str(sorted_indices[1])]
    
    di = cw_combs['c_' + str(sorted_indices[0])][2]
    dj = cw_combs['c_' + str(sorted_indices[1])][2]
    dk = cw_combs['c_' + str(sorted_indices[2])][2]
    
    ci = cw_combs['c_' + str(sorted_indices[0])][0:2]
    cj = cw_combs['c_' + str(sorted_indices[1])][0:2]
    ck = cw_combs['c_' + str(sorted_indices[2])][0:2]
    
    gamma_s = di / np.linalg.norm(r_ij)
    gamma_m = dj / np.linalg.norm(r_ij)
    gamma_l = dk / np.linalg.norm(r_ij)
    
    orientation = determine_order_on_plane(ci, cj, ck)
    
    id_i = id_combs['id_' + str(sorted_indices[0])]
    id_j = id_combs['id_' + str(sorted_indices[1])]
    id_k = id_combs['id_' + str(sorted_indices[2])]
    
    return np.array([int(id_i), int(id_j), int(id_k), alpha_s, alpha_l, gamma_s, gamma_m, gamma_l]), orientation

def determine_orientation(A, B, C, Z):
    # Calculate directional vectors
    AB = B - A
    BC = C - B

    # Calculate the cross product
    N = np.cross(AB, BC)

    # Define the up-direction (Z-axis)
    # Z = np.array([0, 0, 1])

    # Determine the orientation using dot product
    if np.dot(N, Z) > 0:
        return 1 # counter clockwise
    else:
        return -1 # clockwise
    
     
    
def coplanar_triad_descriptors(Ai, Aj, Ak):
    """Generate the seven unique invariants for a triad of coplanar conics.

    Equations (128) through (131) in Christian et al. (2021)

    Args:
        Ai (ArrayLike): Matrix representation of the first conic with dims (...,3,3)
        Aj (ArrayLike): Matrix representation of the second conic
        Ak (ArrayLike): Matrix representation of the third conic

    Returns:
        np.ndarray: The seven invariants with dims (...,7)
    """

    # Normalise matrices such that the determinants are all 1
    Ai = np.cbrt(1 / np.linalg.det(Ai))[..., np.newaxis, np.newaxis] * Ai
    Aj = np.cbrt(1 / np.linalg.det(Aj))[..., np.newaxis, np.newaxis] * Aj
    Ak = np.cbrt(1 / np.linalg.det(Ak))[..., np.newaxis, np.newaxis] * Ak

    Aiinv = np.linalg.inv(Ai)
    Ajinv = np.linalg.inv(Aj)
    Akinv = np.linalg.inv(Ak)

    Iij = np.trace(Aiinv @ Aj, axis1=-2, axis2=-1)
    Iji = np.trace(Ajinv @ Ai, axis1=-2, axis2=-1)
    Iik = np.trace(Aiinv @ Ak, axis1=-2, axis2=-1)
    Iki = np.trace(Akinv @ Ai, axis1=-2, axis2=-1)
    Ijk = np.trace(Ajinv @ Ak, axis1=-2, axis2=-1)
    Ikj = np.trace(Akinv @ Aj, axis1=-2, axis2=-1)

    Aj_plus_Ak = Aj + Ak
    Aj_minus_Ak = Aj - Ak

    Aj_plus_Ak_star = cofactor(Aj_plus_Ak)
    Aj_plus_Ak_star = Aj_plus_Ak_star.T

    Aj_minus_Ak_star = cofactor(Aj_minus_Ak)
    Aj_minus_Ak_star = Aj_minus_Ak_star.T

    # Iijk = np.trace(
    #     (matrix_adjugate(Aj + Ak) - matrix_adjugate(Aj - Ak)) @ Ai, axis1=-2, axis2=-1
    # )
    Iijk = np.trace(
        (Aj_plus_Ak_star - Aj_minus_Ak_star) @ Ai, axis1=-2, axis2=-1
    )

    return np.stack([Iij, Iji, Iik, Iki, Ijk, Ikj, Iijk], axis=-1)