import numpy as np
import healpy as hp
import json
import sys
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations, permutations
import math
from descriptor import *
import pickle
sys.path.append("../crater_pose_estimation-main/")

from src.get_data import *
from src.utils import *
import os
from mpl_toolkits.mplot3d import Axes3D

from numba import njit
from math import acos, sqrt
import random
from joblib import Parallel, delayed
import argparse
import gc

def plot_ellipse(center, major_axis_length, minor_axis_length, angle_degrees, color='blue'):
    """
    Plot an ellipse using matplotlib.

    Args:
    - center (tuple): (x, y) coordinates of the ellipse's center.
    - major_axis_length (float): Length of the major axis.
    - minor_axis_length (float): Length of the minor axis.
    - angle_degrees (float, optional): Orientation angle of the ellipse in degrees. Default is 0.
    - color (str, optional): Color of the ellipse. Default is 'blue'.

    Returns:
    - None
    """
    fig, ax = plt.subplots()

    # Create an ellipse patch
    ellipse = patches.Ellipse(center, major_axis_length, minor_axis_length, angle=angle_degrees, fill=False, edgecolor=color, linewidth=2)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    # Set aspect ratio to equal for accurate representation
    ax.set_aspect('equal', 'box')

    # Set the plot limits
    ax.set_xlim(center[0] - major_axis_length, center[0] + major_axis_length)
    ax.set_ylim(center[1] - major_axis_length, center[1] + major_axis_length)

    plt.show()

def lon_lat_to_xyz(lon, lat, radius):
    """Convert longitudes and latitudes for a body of a given radius to a cartesian
    coordinate system where the +x axis corresponds to longitude zero and +z to the
    North pole i.e. latitude 90 degrees.

    Equation 23 in Christian et al. (2021)
    """

    # has shape (3,...)
    cartesian =  radius * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    cartesian = np.moveaxis(cartesian, 0, -1)  # make shape (..., 3)
    return cartesian


def convert_to_lat_lon(points):
    """
    Convert a set of 3D points to longitude and latitude.

    Parameters:
    - points: numpy array of shape [Mx3] where M is the number of points.

    Returns:
    - lon_lat: numpy array of shape [Mx2] where the first column is longitude and the second is latitude.
    """

    ptsnew = np.hstack((points, np.zeros(points.shape)))
    xy = points[:,0]**2 + points[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + points[:,2]**2)
    # ptsnew[:,4] = np.arctan2(np.sqrt(xy), points[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,4] = np.arctan2(points[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up, lattiude
    ptsnew[:,5] = np.arctan2(points[:,1], points[:,0]) # longitude


    return ptsnew[:, 4:6]

def compute_center_latlon(v1, v2, v3):
    """
    Compute the center of three vectors.
    """
    cx = (v1[0] + v2[0] + v3[0]) / 3
    cy = (v1[1] + v2[1] + v3[1]) / 3
    cz = (v1[2] + v2[2] + v3[2]) / 3

    xy = cx ** 2 + cy ** 2
    lat = np.arctan2(cz, np.sqrt(xy))
    lon = np.arctan2(cy, cx)
    return lat, lon


def angular_distances(points):
    """
    Compute the angular distance between each point to all points in the set.

    Parameters:
    - points: numpy array of shape [Mx3] where M is the number of points.

    Returns:
    - distances: numpy array of shape [MxM] containing angular distances between points.
    """
    # Normalize the points
    norms = np.linalg.norm(points, axis=1)
    normalized_points = points / norms[:, np.newaxis]

    # Compute the dot product between all pairs of points
    dot_products = np.dot(normalized_points, normalized_points.T)

    # Clip values to the range [-1, 1] to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Compute angular distances
    distances = np.rad2deg(np.arccos(dot_products))

    return distances


def haversine_angular_distances(latitudes, longitudes):
    """
    Compute the angular distance between each pair of points given their latitudes and longitudes.

    Parameters:
    - latitudes: numpy array of shape [M] where M is the number of points.
    - longitudes: numpy array of shape [M] where M is the number of points.

    Returns:
    - distances: numpy array of shape [MxM] containing angular distances between points.
    """
    # Convert degrees to radians
    # latitudes = np.radians(latitudes)
    # longitudes = np.radians(longitudes)

    # Compute pairwise differences
    delta_lat = latitudes[:, np.newaxis] - latitudes
    delta_lon = longitudes[:, np.newaxis] - longitudes

    # Haversine formula
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(latitudes[:, np.newaxis]) * np.cos(latitudes) * np.sin(
        delta_lon / 2.0) ** 2
    distances = 2 * np.arcsin(np.sqrt(a))

    return np.rad2deg(distances)

def compute_plane_normal(v1, v2, v3):
    dir1 = v2 - v1
    dir2 = v3 - v1

    # Compute the normal of the plane
    normal = np.cross(dir1, dir2)
    normal = normal / np.linalg.norm(normal)
    return normal

def order_clockwise_3d(v1, v2, v3):
    # Compute direction vectors
    # dir1 = v2 - v1
    # dir2 = v3 - v1
    #
    # # Compute the normal of the plane
    # normal = np.cross(dir1, dir2)
    # normal = normal / np.linalg.norm(normal)
    normal = compute_plane_normal(v1, v2, v3)

    # Use cross product and dot product to determine order of v1 and v2
    cross12 = np.cross(v1, v2)

    vectors = np.array([v1, v2, v3])
    indices = np.array([0, 1, 2])

    if np.dot(cross12, normal) > 0:
        vectors[[0, 1]] = vectors[[1, 0]]
        indices[[0, 1]] = indices[[1, 0]]

    return vectors, indices


def visualize_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b']
    for i, point in enumerate(points):
        ax.quiver(0, 0, 0, point[0], point[1], point[2], color=colors[i], arrow_length_ratio=0.1)

    ax.set_xlim([0, max(p[0] for p in points)+1])
    ax.set_ylim([0, max(p[1] for p in points)+1])
    ax.set_zlim([0, max(p[2] for p in points)+1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def find_closest_entry_NCP(index_dicts_PNCP, v):
    # Extract the values of 'L0', 'L1', and 'L2' into a matrix M
    M = np.array([[entry['L0'], entry['L1'], entry['L2']] for entry in index_dicts_PNCP])

    # Compute the squared distances
    distances_squared = np.sum((M - v) ** 2, axis=1)

    # Find the index with the minimum squared distance
    index = np.argmin(distances_squared)

    # Extract the corresponding dictionary entry
    closest_entry = index_dicts_PNCP[index]

    plt.plot(np.sort(distances_squared), marker='x')
    plt.show()
    # Return the minimum distance (square root of the minimum squared distance), and the values of c1id, c2id, and c3id
    return {
        'distance': np.sqrt(distances_squared[index]),
        'c1id': closest_entry['c1id'],
        'c2id': closest_entry['c2id'],
        'c3id': closest_entry['c3id']
    }

def find_closest_entry_CP(index_dicts_CP, v):
    # Extract the values of 'L0', 'L1', and 'L2' into a matrix M
    M = np.array([[entry['K0'], entry['K1'], entry['K2'], entry['K3'], entry['K4'], entry['K5'], entry['K6']] for entry in index_dicts_CP])

    # Compute the squared distances
    distances_squared = np.sum((M - v) ** 2, axis=1)

    # Find the index with the minimum squared distance
    index = np.argmin(distances_squared)

    # Extract the corresponding dictionary entry
    closest_entry = index_dicts_PNCP[index]

    plt.plot(np.sort(distances_squared), marker='x')
    plt.show()
    # Return the minimum distance (square root of the minimum squared distance), and the values of c1id, c2id, and c3id
    return {
        'distance': np.sqrt(distances_squared[index]),
        'c1id': closest_entry['c1id'],
        'c2id': closest_entry['c2id'],
        'c3id': closest_entry['c3id']
    }

def order_clockwise_2d(p1, p2, p3):
    # Compute the centroid of the points
    centroid = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)

    # Function to compute the angle with respect to the centroid
    def angle_from_centroid(p):
        return math.atan2(p[1] - centroid[1], p[0] - centroid[0])

    # Pair each point with its ID
    points_with_id = [(p1, 1), (p2, 2), (p3, 3)]

    # Sort the points based on their angles
    sorted_points_with_id = sorted(points_with_id, key=lambda x: angle_from_centroid(x[0]), reverse=True)

    # Separate the sorted points and IDs
    sorted_points = [point for point, _ in sorted_points_with_id]
    sorted_ids = [id_ for _, id_ in sorted_points_with_id]

    return np.array(sorted_points), np.array(sorted_ids) - 1

def NCP_descriptors_wrapper(ordered_proj_conics, index_dicts_PNCP, curr_crater_id):
    try:
        descriptor_PNCP_1 = non_coplanar_triad_descriptors(
            ordered_proj_conics[0, ...],
            ordered_proj_conics[1, ...],
            ordered_proj_conics[2, ...]
        )
        # descriptor_PNCP_2 = non_coplanar_triad_descriptors(
        #     ordered_proj_conics[1, ...],
        #     ordered_proj_conics[2, ...],
        #     ordered_proj_conics[0, ...]
        # )
        # descriptor_PNCP_3 = non_coplanar_triad_descriptors(
        #     ordered_proj_conics[2, ...],
        #     ordered_proj_conics[0, ...],
        #     ordered_proj_conics[1, ...]
        # )
        index_dicts_PNCP.append(
            dict(
                c1id=curr_crater_id[0],
                c2id=curr_crater_id[1],
                c3id=curr_crater_id[2],
                L0=descriptor_PNCP_1[0],
                L1=descriptor_PNCP_1[1],
                L2=descriptor_PNCP_1[2],
            )
        )
        # index_dicts_PNCP.append(
        #     dict(
        #         c1id=curr_crater_id[1],
        #         c2id=curr_crater_id[2],
        #         c3id=curr_crater_id[0],
        #         L0=descriptor_PNCP_2[0],
        #         L1=descriptor_PNCP_2[1],
        #         L2=descriptor_PNCP_2[2],
        #     )
        # )
        # index_dicts_PNCP.append(
        #     dict(
        #         c1id=curr_crater_id[2],
        #         c2id=curr_crater_id[0],
        #         c3id=curr_crater_id[1],
        #         L0=descriptor_PNCP_3[0],
        #         L1=descriptor_PNCP_3[1],
        #         L2=descriptor_PNCP_3[2],
        #     )
        # )
    except ValueError:
        pass

    return index_dicts_PNCP

def CP_descriptors_wrapper(ordered_proj_conics, curr_crater_id):
    index_dicts_PCP = []
    try:
        descriptor_PCP_1 = coplanar_triad_descriptors(
            ordered_proj_conics[0, ...],
            ordered_proj_conics[1, ...],
            ordered_proj_conics[2, ...],
        )

        index_dicts_PCP.append(
            dict(
                c1id=curr_crater_id[0],
                c2id=curr_crater_id[1],
                c3id=curr_crater_id[2],
                K0=descriptor_PCP_1[0],
                K1=descriptor_PCP_1[1],
                K2=descriptor_PCP_1[2],
                K3=descriptor_PCP_1[3],
                K4=descriptor_PCP_1[4],
                K5=descriptor_PCP_1[5],
                K6=descriptor_PCP_1[6],
            )
        )
    except ValueError:
        pass
    return index_dicts_PCP

def angular_distance_matrix(v1, v2, v3):
    vectors = [v1, v2, v3]
    n = len(vectors)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dot_product = np.dot(vectors[i], vectors[j])
                magnitude_i = np.linalg.norm(vectors[i])
                magnitude_j = np.linalg.norm(vectors[j])
                cos_theta = dot_product / (magnitude_i * magnitude_j)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle potential numerical errors
                matrix[i][j] = np.rad2deg(theta)

    return matrix

def random_3d_unit_vector(mag):
    # Generate random x, y, z components
    x, y, z = np.random.rand(3) - 0.5  # Subtracting 0.5 to make the range [-0.5, 0.5]

    # Compute the magnitude of the vector
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Normalize the vector
    x /= magnitude
    y /= magnitude
    z /= magnitude

    return np.array([x, y, z]) * mag


def find_furthest_points(angular_distance_matrix):
    """
    For each 3D point P1 (represented by its index), find the index of P2 that is furthest away
    and the index of P3 that has the largest average angular distance from P1 and P2.

    :param angular_distance_matrix: np.ndarray, pairwise angular distance matrix.
    :return: List[Tuple[int, int, int]], list of tuples containing indices of P1, P2, and P3 for each 3D point.
    """
    result = []
    n = len(angular_distance_matrix)

    for i in range(n):
        # Find the index of the point P2 that is furthest away from P1
        p1_index = i
        p2_index = np.argmax(angular_distance_matrix[i])

        # Find the index of the point P3 that has the largest average angular distance from P1 and P2
        max_avg_distance = 0
        p3_index = -1
        for j in range(n):
            if j == p1_index or j == p2_index:
                continue
            avg_distance = (angular_distance_matrix[p1_index, j] + angular_distance_matrix[p2_index, j]) / 2
            if avg_distance > max_avg_distance:
                max_avg_distance = avg_distance
                p3_index = j

        result.append((p1_index, p2_index, p3_index))

    return result


def find_closest_points(angular_distance_matrix):
    """
    For each 3D point P1 (represented by its index), find the index of P2 that is closest
    and the index of P3 that has the smallest average angular distance from P1 and P2.

    :param angular_distance_matrix: np.ndarray, pairwise angular distance matrix.
    :return: List[Tuple[int, int, int]], list of tuples containing indices of P1, P2, and P3 for each 3D point.
    """
    result = []
    n = len(angular_distance_matrix)

    for i in range(n):
        # Find the index of the point P2 that is closest to P1
        p1_index = i
        # Set the diagonal to infinity to avoid selecting P1 as P2
        np.fill_diagonal(angular_distance_matrix, np.inf)
        p2_index = np.argmin(angular_distance_matrix[i])

        # Find the index of the point P3 that has the smallest average angular distance from P1 and P2
        min_avg_distance = np.inf
        p3_index = -1
        for j in range(n):
            if j == p1_index or j == p2_index:
                continue
            avg_distance = (angular_distance_matrix[p1_index, j] + angular_distance_matrix[p2_index, j]) / 2
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                p3_index = j

        result.append((p1_index, p2_index, p3_index))

    return result
@njit
def angular_distance(p1, p2):
    dot_product = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
    magnitude_p1 = sqrt(p1[0]**2 + p1[1]**2 + p1[2]**2)
    magnitude_p2 = sqrt(p2[0]**2 + p2[1]**2 + p2[2]**2)
    cos_theta = dot_product / (magnitude_p1 * magnitude_p2)
    # Clamp the value between -1 and 1 to avoid numerical errors
    cos_theta = max(min(cos_theta, 1), -1)
    return acos(cos_theta)

@njit
def compute_angular_distance_matrix(points):
    n = len(points)
    angular_distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance = angular_distance(points[i], points[j])
            angular_distance_matrix[i, j] = distance
            angular_distance_matrix[j, i] = distance
    return angular_distance_matrix


@njit
def compute_pairwise_euclidean_distance(points):
    """
    Computes the pairwise Euclidean distance matrix for a set of 2D points.

    :param points: np.ndarray, array of 2D points.
    :return: np.ndarray, pairwise Euclidean distance matrix.
    """
    n = len(points)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # The distance matrix is symmetric

    return distance_matrix


def find_largest_avg_distance_triad(distance_matrix):
    """
    Finds the triad of points (P1, P2, P3) that has the largest average distance between P1 and P2, P2 and P3, and P1 and P3.

    :param distance_matrix: np.ndarray, pairwise distance matrix.
    :return: Tuple[int, int, int], indices of P1, P2, and P3 forming the triad with the largest average distance.
    """
    n = len(distance_matrix)
    max_avg_distance = 0
    selected_triad = (-1, -1, -1)

    for i, j, k in combinations(range(n), 3):
        avg_distance = (distance_matrix[i, j] + distance_matrix[j, k] + distance_matrix[k, i]) / 3
        if avg_distance > max_avg_distance:
            max_avg_distance = avg_distance
            selected_triad = (i, j, k)

    return selected_triad


@njit
def find_closest_point_id(X, M):
    """
    Finds the ID (index) of the point in M that has the smallest Euclidean distance to X.

    :param X: np.ndarray, the 3D point.
    :param M: np.ndarray, the set of 3D points.
    :return: int, the ID (index) of the closest point in M to X.
    """
    min_distance = float('inf')
    closest_id = -1
    n = len(M)

    for i in range(n):
        distance = np.sqrt((X[0] - M[i, 0]) ** 2 + (X[1] - M[i, 1]) ** 2 + (X[2] - M[i, 2]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_id = i

    return closest_id

def write_values_to_file_global(index_dicts_PCP, filename):
    with open(filename, 'a') as file:
        for d in index_dicts_PCP:
            # Extract values and keep first 3 unchanged
            unchanged_values = list(d.values())[:3]
            # Round the rest
            rounded_values = [round(float(x), 4) for x in list(d.values())[3:]]
            all_values = unchanged_values + rounded_values
            values_str = ' '.join(map(str, all_values))
            file.write(values_str + '\n')

def write_values_to_file_local(index_dicts_PCP, filename):
    with open(filename, 'a') as file:
        for d in index_dicts_PCP:
            # Extract values and keep first 3 unchanged
            unchanged_values = list(d.values())[:3]
            # Round the rest
            rounded_values = [round(float(x), 2) for x in list(d.values())[3:]]
            all_values = unchanged_values + rounded_values
            values_str = ' '.join(map(str, all_values))
            file.write(values_str + '\n')

def process_crater_id(curr_crater_id, local_craters, local_conics, local_conics_inv, local_Hmi_K, local_crater_idx, local_ids, centre_ipix, nside):
    # Logic from inside your loop
    # ... existing logic ...

    triad_craters = local_craters[np.isin(local_crater_idx, [curr_crater_id])]
    triad_conics = local_conics[np.isin(local_crater_idx, [curr_crater_id])]
    triad_conics_inv = local_conics_inv[np.isin(local_crater_idx, [curr_crater_id])]
    triad_Hmi_K = local_Hmi_K[np.isin(local_crater_idx, [curr_crater_id])]
    triad_ids = local_ids[np.isin(local_crater_idx, [curr_crater_id])]

    # determine the centroid of the triad
    cent_lat, cent_lon = compute_center_latlon(triad_craters[0, 0:3], triad_craters[1, 0:3], triad_craters[2, 0:3])

    # if it does not belong to the centre_pix, move on
    triad_centre_ipix = hp.ang2pix(nside, theta=np.rad2deg(cent_lon), phi=np.rad2deg(cent_lat), lonlat=True)
    index_dicts_PCP = []
    
    if triad_centre_ipix == centre_ipix:
        plane_normal = compute_plane_normal(triad_craters[0, 0:3], triad_craters[1, 0:3], triad_craters[2, 0:3])
        looking_at_cam, _ = create_extrinsic_matrix(plane_normal, np.linalg.norm(triad_craters[0, 0:3]) + 100000)

        # project conics
        proj_triad_conics = np.zeros_like(triad_conics_inv)
        proj_triad_params = np.zeros([3, 6])
        for tci in range(triad_conics_inv.shape[0]):
            proj_triad_conics[tci] = conic_from_crater_cpu(triad_conics_inv[tci], triad_Hmi_K[tci], looking_at_cam)
            proj_triad_params[tci] = extract_ellipse_parameters_from_conic(proj_triad_conics[tci])

        # Otherwise, stay on, order the triads
        # ordered_vectors, ordered_indices = order_clockwise_2d(proj_triad_params[0, 1:3], proj_triad_params[1, 1:3],
        #                                                       proj_triad_params[2, 1:3])

        # ordered_proj_conics = proj_triad_conics[ordered_indices]
        # ordered_triad_ids = triad_ids[ordered_indices]

        index_dicts_PCP = CP_descriptors_wrapper(proj_triad_conics, triad_ids)
    # Return the necessary results
    return index_dicts_PCP  # Replace with actual return value

def main_func_local(centre_ipix, nside, total_side, db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_Hmi_k, ID):
    # for centre_ipix in range(start_id, end_id):
        # for centre_ipix in range(1):
    print('Progress: ' + str(centre_ipix) + ' / ' + str(total_side)  + '\n')
    num_skips = 0

    # For each pixel, consider each triad with all three craters in the 3x3 pixel
    # area centred around the current pixel
    # For each pixel, consider each triad with all three craters in the 3x3 pixel
    # area centred around the current pixel
    neighbours = np.append(hp.get_all_neighbours(nside, centre_ipix), centre_ipix)

    local_craters = db_CW_params[np.isin(ipix, neighbours)]
    local_conics = db_CW_conic[np.isin(ipix, neighbours)]
    local_conics_inv = db_CW_conic_inv[np.isin(ipix, neighbours)]
    local_Hmi_K = db_CW_Hmi_k[np.isin(ipix, neighbours)]
    local_ids = ID[np.isin(ipix, neighbours)]

    results = []
    if not(local_craters.shape[0] == 0):
        local_crater_idx = idx[np.isin(ipix, neighbours)]
        local_crater_id_triads = combinations(local_crater_idx, 3)
        # local_crater_id_triads = permutations(local_crater_idx, 3)

        # Parallel execution
        results = Parallel(n_jobs=-1)(
            delayed(process_crater_id)(
                curr_crater_id, local_craters, local_conics, local_conics_inv, local_Hmi_K, local_crater_idx,
                local_ids, centre_ipix, nside
            ) for curr_crater_id in local_crater_id_triads
        )

    return results

import time

if __name__ == "__main__":
    # home_dir = '/data/Dropbox/craters/christian_craters_ID/data/'
    # parser = argparse.ArgumentParser(description="Script to process data.")
    # parser.add_argument("--data_dir", required=True, help="data_dir")
    # parser.add_argument("--out_dir", required=True, help="out_dir")
    # parser.add_argument("--time_dir", required=True, help="out_dir")
    # parser.add_argument("--nside", type=int, required=True, help="nside")
    # # parser.add_argument("--database_type", required=True, help="database_type")
    # parser.add_argument("--all_craters_world_dir", required=True, help="all_craters_world_dir")
    
    # args = parser.parse_args()

    # data_dir = args.data_dir
    # out_dir = args.out_dir
    # time_dir = args.time_dir
    # nside = args.nside
    # # database_type = args.database_type
    # all_craters_world_dir = args.all_craters_world_dir

    # save_dir = out_dir + '/local_coplanar_' + str(nside)

    # calibration_file = data_dir + '/calibration.txt'

    base_dir = os.getcwd()
    data_dir = base_dir + '/data'
    out_dir = base_dir + '/data/descriptor_db/pkl'
    all_craters_world_dir = base_dir + '/data/descriptor_db/filtered_catalog.txt'

    calibration_file = data_dir + '/calibration.pkl'

    nside = 32

    save_dir = out_dir + '/local_coplanar_' + str(nside)

    start_id = 0

    # Get the camera intrinsic matrix and distortion coeffients.
    K = get_intrinsic(calibration_file)

    # create a full dataset dir
    with open(all_craters_world_dir, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
    lines = [i.split(',') for i in lines]
    lines = np.array(lines)

    ID = lines[:, 0]
    lines = np.float64(lines[:, 1:])
    
    # convert all to conics
    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k = get_craters_world_numba(lines)

    # Identify indices of repetitive elements in ID
    unique_ID, indices, counts = np.unique(ID, return_index=True, return_counts=True)
    removed_indices = np.setdiff1d(np.arange(ID.shape[0]), indices)

    # Remove rows from matrices
    db_CW_params = np.delete(db_CW_params, removed_indices, axis=0)
    db_CW_conic = np.delete(db_CW_conic, removed_indices, axis=0)
    db_CW_conic_inv = np.delete(db_CW_conic_inv, removed_indices, axis=0)
    db_CW_ENU = np.delete(db_CW_ENU, removed_indices, axis=0)
    db_CW_Hmi_k = np.delete(db_CW_Hmi_k, removed_indices, axis=0)
    ID = np.delete(ID, removed_indices, axis=0)

    idx = np.array(range(0, db_CW_params.shape[0]))

    lat_lon = convert_to_lat_lon(db_CW_params[:, 0:3])

    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, theta=np.rad2deg(lat_lon[:, 1]), phi=np.rad2deg(lat_lon[:, 0]), lonlat=True)

    ipix_id = 0

    index_dicts_PCP = []
	

    # Since the result file might be too big, hence I save them at a fixed interval to prevent out of RAM problem and
    # Run db_compilation to combine all into one file 
    for centre_ipix in range(hp.nside2npix(nside)):
        curr_dicts = main_func_local(centre_ipix, nside, NPIX, db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_Hmi_k, ID)
        filtered_curr_dicts = [sublist for sublist in curr_dicts if sublist]
        index_dicts_PCP.append(filtered_curr_dicts)

        # Save at every 100 interval
        interval = 100 
        if centre_ipix % interval == 0 and centre_ipix != 0:
            save_path = f"{save_dir}_part_{centre_ipix // interval}.pkl"
            with open(save_path, 'wb') as file:
                pickle.dump(index_dicts_PCP, file)
            index_dicts_PCP = []
            print(f"Saved data up to index {centre_ipix}")

    # Save the final batch if it's not empty
    if index_dicts_PCP:
        save_path = f"{save_dir}_final.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(index_dicts_PCP, file)
        print("Saved final batch of data")

