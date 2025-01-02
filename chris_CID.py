from itertools import combinations, permutations
from descriptor import *

from src.get_data import *
from src.utils import *
from joblib import Parallel, delayed
from numba import njit
from scipy.spatial import cKDTree
import csv
import argparse
import time
import pickle
import os


@njit
def gaussian_angle(Ai_params, Aj_params):
    xc_i, yc_i, a_i, b_i, phi_i = Ai_params
    xc_j, yc_j, a_j, b_j, phi_j = Aj_params

    y_i = np.array([xc_i, yc_i])
    y_j = np.array([xc_j, yc_j])

    Yi_phi = np.array([[np.cos(phi_i), -np.sin(phi_i)], [np.sin(phi_i), np.cos(phi_i)]])
    Yj_phi = np.array([[np.cos(phi_j), -np.sin(phi_j)], [np.sin(phi_j), np.cos(phi_j)]])

    Yi_len = np.array([[1 / a_i ** 2, 0], [0, 1 / b_i ** 2]])
    Yj_len = np.array([[1 / a_j ** 2, 0], [0, 1 / b_j ** 2]])

    Yi_phi_t = np.transpose(Yi_phi)
    Yj_phi_t = np.transpose(Yj_phi)

    Yi = np.dot(Yi_phi, np.dot(Yi_len, Yi_phi_t))
    Yj = np.dot(Yj_phi, np.dot(Yj_len, Yj_phi_t))

    Yi_det = np.linalg.det(Yi)
    Yj_det = np.linalg.det(Yj)

    # Compute the difference between the vectors
    diff = y_i - y_j

    # Compute the sum of the matrices
    Y_sum = Yi + Yj

    # Invert the resulting matrix
    Y_inv = np.linalg.inv(Y_sum)

    # Compute the expression
    exp_part = np.exp(-0.5 * diff.T @ Yi @ Y_inv @ Yj @ diff)

    front_part = (4 * np.sqrt(Yi_det * Yj_det)) / np.linalg.det(Y_sum)

    dGA = np.arccos(np.minimum(front_part * exp_part, 1))
    return dGA ** 2


def strip_symbols(s, symbols):
    for symbol in symbols:
        s = s.replace(symbol, '')
    return s


def testing_data_read_image_params(dir):
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)
    
    noisy_imaged_params = []
    image_names = []

    for row_id, row in enumerate(data):
        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[3])]
        noisy_imaged_params.append(curr_imaged_params)
    
    for row_id, row in enumerate(data):
        # Extract Image names 
        image_names.append(row[11])
    
    return noisy_imaged_params, image_names


def testing_data_reading_general(dir):
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)

    camera_extrinsic = np.zeros([len(data), 3, 4])
    camera_pointing_angle = np.zeros(len(data))
    heights = np.zeros(len(data))
    noise_levels = np.zeros(len(data))
    remove_percentages = np.zeros(len(data))
    add_percentages = np.zeros(len(data))
    att_noises = np.zeros(len(data))  # Att_noise is always one value
    noisy_cam_orientations = np.zeros([len(data), 3, 3])  # Noisy cam orientation is always a 3x3 matrix

    imaged_params = []
    noisy_imaged_params = []
    crater_indices = []

    for row_id, row in enumerate(data):
        # Extract Camera Extrinsic matrix
        row_0 = row[0].split('\n')
        curr_cam_ext = np.zeros([3, 4])
        for i in range(len(row_0)):
            curr_row = strip_symbols(row_0[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 4)
            curr_cam_ext[i] = curr_array
        camera_extrinsic[row_id] = curr_cam_ext

        # Extract Camera Pointing Angle
        camera_pointing_angle[row_id] = float(row[1])

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[2])]
        imaged_params.append(curr_imaged_params)

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[3])]
        noisy_imaged_params.append(curr_imaged_params)

        # Extract Crater Indices
        # crater_indices.append(literal_eval(row[4]))
        curr_conic_indices = np.array(eval(row[4]))
        crater_indices.append(curr_conic_indices)

        # Extract Height
        heights[row_id] = float(row[5])

        # Extract Noise Level
        noise_levels[row_id] = float(row[6])

        # Extract Remove Percentage
        remove_percentages[row_id] = float(row[7])

        # Extract Add Percentage
        add_percentages[row_id] = float(row[8])

        # Extract Attitude Noise
        att_noises[row_id] = float(row[9])

        # Extract Noisy Camera Orientation
        # noisy_cam_orientations[row_id] = np.array(literal_eval(row[10]))
        row_10 = row[10].split('\n')
        curr_nc = np.zeros([3, 3])
        for i in range(len(row_10)):
            curr_row = strip_symbols(row_10[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 3)
            curr_nc[i] = curr_array
        noisy_cam_orientations[row_id] = curr_nc

    return camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, crater_indices, \
           heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations



def testing_data_reading(dir):
    # Read the CSV file
    # Read the CSV file
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Skip headers
    data = data[1:]
    camera_extrinsic = np.zeros([len(data), 3, 4])
    camera_pointing_angle = np.zeros(len(data))
    imaged_params = []
    noisy_imaged_params = []
    conic_indices = []

    for row_id, row in enumerate(data):
        # Extract Camera Extrinsic matrix
        row_0 = row[0].split('\n')
        curr_cam_ext = np.zeros([3, 4])
        for i in range(len(row_0)):
            curr_row = strip_symbols(row_0[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 4)
            curr_cam_ext[i] = curr_array

        camera_extrinsic[row_id] = curr_cam_ext
        # Extract Camera Pointing Angle
        camera_pointing_angle[row_id] = float(row[1])

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[2])]
        imaged_params.append(curr_imaged_params)

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[3])]
        noisy_imaged_params.append(curr_imaged_params)

        # Extract Conic Indices
        curr_conic_indices = np.array(eval(row[4]))
        conic_indices.append(curr_conic_indices)

    return camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, conic_indices


def NLLS_rm(CC_params, CC_conic, CW_params, CW_conic, gt_att, CW_ENU, K):
    '''
    Args:
        CC_params:
        CC_conic:
        CW_params:
        CW_conic:
        gt_att: Known camera orientation
        CW_ENU: crater's ENU coordinates
        K: intrinsic parameters

    Returns:

    '''
    S = np.array([[1, 0], [0, 1], [0, 0]])
    k = np.array([0, 0, 1])
    k = k[:, np.newaxis]
    A_stack = []
    b_stack = []
    for i in range(CC_params.shape[0]):
        Ci = CW_conic[i]  # Ci = conic from craters map
        Ai = CC_conic[i]  # Ai = imaged conics
        Pm_i = CW_params[i, 0:3]
        T_M_C = gt_att  # T^M_C = inverse of camera's orientation in Moon's coordinate
        T_E_M = CW_ENU[i]  # T^E_M = [e, n, u]

        B = T_M_C.T @ K.T @ Ai @ K @ T_M_C  # B = T^C_M K.T A K T^M_c
        SCS = S.T @ Ci @ S
        STBTS = S.T @ T_E_M.T @ B @ T_E_M @ S

        STBTS_flat = STBTS.flatten()
        SCS_flat = SCS.flatten()
        numerator = np.dot(SCS_flat.T, STBTS_flat)
        denominator = np.dot(SCS_flat.T, SCS_flat)
        s_i = numerator / denominator

        A_stack.append(S.T @ T_E_M.T @ B)
        STBp = S.T @ T_E_M.T @ B @ Pm_i
        sSCk = s_i * S.T @ Ci @ k
        b_stack.append(STBp[:, np.newaxis] - sSCk)

    A_mat = np.vstack(A_stack)
    B_mat = np.vstack(b_stack)
    rm = np.linalg.pinv(A_mat) @ B_mat
    return rm


def find_indices(query, ID):
    indices = []
    for q in query:
        idx = np.where(ID == q)[0]
        if idx.size > 0:
            indices.append(idx[0])
    return np.array(indices)


def test_find_full_match_indices():
    # Test Case 1: Typical case
    query = ["a", "b"]
    ID = np.array([["a", "b"], ["b", "a"], ["a", "c"]])
    expected = np.array([0, 1])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 1 Failed"

    # Test Case 2: No match
    query = ["x", "y"]
    ID = np.array([["a", "b"], ["b", "a"], ["a", "c"]])
    expected = np.array([])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 2 Failed"

    # Test Case 3: Empty query
    query = []
    ID = np.array([["a", "b"], ["b", "a"], ["a", "c"]])
    expected = np.array([])  # Assuming no match for empty query
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 3 Failed"

    # Test Case 4: Empty ID
    query = ["a", "b"]
    ID = np.array([])
    expected = np.array([])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 4 Failed"

    # Test Case 5: Query longer than any row in ID
    query = ["a", "b", "c"]
    ID = np.array([["a", "b"], ["b", "a"], ["a"]])
    expected = np.array([])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 5 Failed"

    # Test Case 6: ID with rows of varying lengths
    query = ["a", "b"]
    ID = np.array([["a"], ["a", "b"], ["b", "a", "c"], ["a", "b"]])
    expected = np.array([1, 3])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 6 Failed"

    # Test Case 7: Query with duplicate elements
    query = ["a", "a"]
    ID = np.array([["a", "a"], ["a", "b"], ["b", "a"]])
    expected = np.array([0])
    assert np.array_equal(find_full_match_indices(query, ID), expected), "Test Case 7 Failed"

    print("All test cases passed!")


@njit
def find_full_match_indices(query, ID):
    """
    Finds the indices of rows in the 2D array ID where each row contains all elements of the 1D array query,
    regardless of the order of elements.

    Parameters:
    - query: 1D array of strings, representing the query elements.
    - ID: 2D array of strings, representing the array to be searched.

    Returns:
    - A numpy array of indices of the rows in ID that fully match the query.
    """
    indices = []
    query_len = len(query)

    for idx in range(ID.shape[0]):
        row = ID[idx]
        if len(row) != query_len:
            continue  # Skip rows that do not have the same number of elements as the query

        match = True
        for q in query:
            found = False
            for r in row:
                if q == r:
                    found = True
                    break
            if not found:
                match = False
                break

        if match:
            indices.append(idx)
            return np.array(indices)

    return np.array(indices)



@njit
def compute_pairwise_ga(remaining_CC_params, neighbouring_craters_id, db_CW_conic, db_CW_Hmi_k, curr_cam):
    pairwise_ga = np.ones((remaining_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    for ncid in range(len(neighbouring_craters_id)):
        A = conic_from_crater_cpu_mod(db_CW_conic[neighbouring_craters_id[ncid]],
                                      db_CW_Hmi_k[neighbouring_craters_id[ncid]],
                                      curr_cam)  # project them onto the camera
        # convert A to ellipse parameters
        flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)
        # compute ga with all imaged conics
        if np.any(np.isnan([x_c, y_c, a, b, phi])):
            continue
        if flag:
            for cc_id in range(remaining_CC_params.shape[0]):
                pairwise_ga[cc_id, ncid] = gaussian_angle(remaining_CC_params[cc_id],
                                                          [x_c, y_c, a, b, phi])  # measure pairwise GA

    return pairwise_ga


def descriptor_distance(des_tree, descriptors, gt_indexes, top_n=10):
    distances, q_indexes = des_tree.query(descriptors, k=top_n)

    # Initialize a list to store the rankings
    rankings = []

    # Iterate through the ground truth indexes
    for gt_index in gt_indexes:
        # Iterate through the query indexes and check for matches
        for i, neighbors in enumerate(q_indexes):
            if gt_index in neighbors:
                # Append the ranking (1-indexed) of the gt_index in the neighbors list
                rankings.append(list(neighbors).index(gt_index) + 1)
                break
        else:
            # If no match is found, append a ranking higher than top_n
            rankings.append(11)

    # Return the rankings
    return rankings


def reprojection_test(des_tree, descriptors, c_ids, ID,
                      db_CW_conic, db_CW_Hmi_k, db_CW_params, db_CW_ENU, K, gt_att,
                      ordered_CC_params, ordered_CC_conics, ordered_sigma_sqr,
                      comb_id, matched_ids):
    distances, indexes = des_tree.query(descriptors, k=3)

    # pick N best descriptors
    est_cam = np.zeros([3, 4])
    rm = np.zeros([3])
    for index in indexes:
        craters_id = c_ids[index]

        # convert IDS to unique IDS
        matched_idx = find_indices(craters_id, ID)

        # get from db_CW_Conic
        curr_CW_Conic = db_CW_conic[matched_idx]
        curr_CW_Hmi_k = db_CW_Hmi_k[matched_idx]
        curr_CW_params = db_CW_params[matched_idx]
        curr_CW_ENU = db_CW_ENU[matched_idx]

        # NLLS
        rm = NLLS_rm(ordered_CC_params, ordered_CC_conics, curr_CW_params, curr_CW_Conic, gt_att, curr_CW_ENU, K)
        rm = np.squeeze(rm)
        est_cam = np.zeros([3, 4])
        est_cam[0:3, 0:3] = gt_att
        est_cam[0:3, 3] = -gt_att @ rm
        # project all craters centers to image plane
        est_cam = K @ est_cam

        # project the current estimated craters
        matched = np.zeros(3)
        for j in range(3):
            A = conic_from_crater_cpu_mod(curr_CW_Conic[j], curr_CW_Hmi_k[j], est_cam)

            # convert A to ellipse parameters
            flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)

            if np.any(np.isnan([x_c, y_c, a, b, phi])):
                continue

            if (flag):  # if it's proper conic
                curr_ga = gaussian_angle(ordered_CC_params[j], [x_c, y_c, a, b, phi])
                if ((curr_ga / ordered_sigma_sqr[j]) <= 13.277):
                    matched[j] = 1

        if np.sum(matched) == 3:
            for j in range(3):
                matched_ids[comb_id[j]] = craters_id[j]
            return True, est_cam, rm, matched_idx, matched_ids

    return False, est_cam, rm, matched_idx, matched_ids


def read_descriptors(filename):
    ids = []  # List to store the first 3 values of each line
    values = []  # List to store the remaining values of each line

    with open(filename, 'r') as file:
        for line in file:
            try:
                curr_values = line.strip().split()
                ids.append(curr_values[:3])  # First 3 values as strings
                values.append([float(x) for x in curr_values[3:]])  # Remaining values as floats
            except:
                continue

    # Convert lists to Numpy arrays
    ids = np.array(ids)
    values = np.array(values)

    return ids, values


def read_crater_database(craters_database_text_dir):
    with open(craters_database_text_dir, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
    lines = [i.split(',') for i in lines]
    lines = np.array(lines)

    ID = lines[:, 0]
    lines = np.float64(lines[:, 1:])

    # convert all to conics
    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k = get_craters_world_numba(lines)

    # remove craters
    # Read the file and store IDs in a list
    # with open(to_be_removed_dir, 'r') as file:
    #     removed_ids = [line.strip() for line in file.readlines()]

    # # Find the indices of these IDs in the ID array
    # removed_indices = np.where(np.isin(ID, removed_ids))[0]

    # Remove the craters with indices in removed_indices from your data arrays
    # db_CW_params = np.delete(db_CW_params, removed_indices, axis=0)
    # db_CW_conic = np.delete(db_CW_conic, removed_indices, axis=0)
    # db_CW_conic_inv = np.delete(db_CW_conic_inv, removed_indices, axis=0)
    # db_CW_ENU = np.delete(db_CW_ENU, removed_indices, axis=0)
    # db_CW_Hmi_k = np.delete(db_CW_Hmi_k, removed_indices, axis=0)
    # ID = np.delete(ID, removed_indices, axis=0)

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

    crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])

    return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree


def find_nearest_neighbors(dist_matrix):
    M, N = dist_matrix.shape

    # Placeholder for nearest neighbors for each row
    nearest_neighbors_id = -np.ones(M, dtype=np.int32)
    nearest_neighbors_val = np.ones(M, dtype=np.float32) * np.inf

    # Flatten and argsort manually
    flat_size = M * N
    flat_distances = np.empty(flat_size, dtype=dist_matrix.dtype)
    for i in range(M):
        for j in range(N):
            flat_distances[i * N + j] = dist_matrix[i, j]

    sorted_indices = np.argsort(flat_distances)
    assigned_columns = set()
    for k in range(flat_size):
        index = sorted_indices[k]
        i = index // N
        j = index % N

        if nearest_neighbors_id[i] == -1 and j not in assigned_columns:
            nearest_neighbors_id[i] = j
            nearest_neighbors_val[i] = dist_matrix[i, j]
            assigned_columns.add(j)

        # Break when all rows have been assigned
        if not np.any(nearest_neighbors_id == -1):
            break

    return nearest_neighbors_id, nearest_neighbors_val


def visible_points_on_sphere(points, sphere_center, sphere_radius, camera_position, valid_indices):
    """Return the subset of the 3D points on the sphere that are visible to the camera."""
    visible_points = []
    visible_indices = []
    visible_len_P_cam = []
    non_visible_len_P_cam = []

    for idx in valid_indices:
        point = points[idx, :]

        # 1. Translate the origin to the camera
        P_cam = point - camera_position

        # 2. Normalize the translated point
        P_normalized = P_cam / np.linalg.norm(P_cam)

        # 3 & 4. Solve for the real roots
        # Coefficients for the quadratic equation
        a = np.dot(P_normalized, P_normalized)
        b = 2 * np.dot(P_normalized, camera_position - sphere_center)
        c = np.dot(camera_position - sphere_center, camera_position - sphere_center) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

        min_root = np.minimum(root1, root2)
        # 5. Check which real root matches the length of P_cam
        length_P_cam = np.linalg.norm(P_cam)

        # 6 & 7. Check visibility
        if (np.abs(min_root - length_P_cam) < 1000):
            visible_points.append(point)
            visible_indices.append(idx)
            visible_len_P_cam.append(length_P_cam)
        else:
            # non_visible_points.append(point)
            non_visible_len_P_cam.append(length_P_cam)

    # 4) impose a check that we didnt eliminate points that are within the visible region because of a sub-optimal thresholding above
    #         # compute min and max distance for the visible_pts with the camera,
    #         # if there are other points that are within that range, raise a flag
    if len(non_visible_len_P_cam) > 0 and len(visible_len_P_cam) > 0:
        if np.min(np.array(non_visible_len_P_cam)) < np.max(np.array(visible_len_P_cam)):
            print('Something is wrong\n')

    return visible_points, visible_indices


def convert_to_int(array):
    """
    Converts an array of strings in the format '01-01-000762' to integers by removing hyphens.

    Parameters:
    - array: NumPy array of strings.

    Returns:
    - NumPy array of integers.
    """
    return np.array([int(item.replace('-', '')) for item in array])


def lists_iterating(loaded_array, type):
    cid = []
    descriptors = []
    for outer_layer in loaded_array:
        for middle_layer in outer_layer:
            for inner_dict in middle_layer:
                # Extract the elements if they exist in the dictionary
                c1id = inner_dict.get('c1id')
                c2id = inner_dict.get('c2id')
                c3id = inner_dict.get('c3id')

                # You can choose to store these in a tuple or another structure
                if type == 'local':
                    K0 = inner_dict.get('K0')
                    K1 = inner_dict.get('K1')
                    K2 = inner_dict.get('K2')
                    K3 = inner_dict.get('K3')
                    K4 = inner_dict.get('K4')
                    K5 = inner_dict.get('K5')
                    K6 = inner_dict.get('K6')

                    if np.any(np.isnan([K0, K1, K2, K3, K4, K5, K6])):
                        continue

                    descriptors.append((K0, K1, K2, K3, K4, K5, K6))
                elif type == 'global':
                    L0 = inner_dict.get('L0')
                    L1 = inner_dict.get('L1')
                    L2 = inner_dict.get('L2')

                    if np.any(np.isnan([L0, L1, L2])):
                        continue

                    descriptors.append((L0, L1, L2))
                cid.append((c1id, c2id, c3id))
    return cid, descriptors


def descriptor_reading_pkl(main_dir, prefix, type):
    # Iterate over all files in the directory
    cid = []
    descriptors = []
    for filename in os.listdir(main_dir):
        # Check if the filename starts with your desired prefix
        if filename.startswith(prefix):
            # Full path to the file
            filepath = os.path.join(main_dir, filename)

            with open(filepath, 'rb') as file:
                loaded_array = pickle.load(file)

                for arr_id, sub_arr in enumerate(loaded_array):
                    filtered_arr = [sublist for sublist in sub_arr if sublist]
                    loaded_array[arr_id] = filtered_arr

                curr_cid, curr_descriptors = lists_iterating(loaded_array, type)
                cid.append(np.array(curr_cid))
                descriptors.append(np.array(curr_descriptors))

    cid = np.vstack(cid)
    descriptors = np.vstack(descriptors)

    return cid, descriptors


def get_sort_order(arr1, arr2):
    """
    Gets the order of indices to sort arr1 to match the order of arr2.

    Parameters:
    arr1 (list of str): The array to be sorted.
    arr2 (list of str): The array to match.

    Returns:
    list of int: The order of indices to sort arr1.
    """

    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length.")

    # Create a dictionary to map the elements of arr2 to their indices
    index_map = {value: index for index, value in enumerate(arr2)}

    # Get the order of indices for sorting arr1
    try:
        order = [index_map[x] for x in arr1]
    except KeyError as e:
        raise ValueError(f"Element {e} in arr1 does not exist in arr2") from e

    return order


def compute_thetas(craters_center_point):
    thetas = np.zeros(3)
    normed_craters_center_point = [curr_craters_center_point / np.linalg.norm(curr_craters_center_point) for
                                   curr_craters_center_point in craters_center_point]
    thetas[0] = np.rad2deg(np.arccos(np.dot(normed_craters_center_point[0], normed_craters_center_point[1])))
    thetas[1] = np.rad2deg(np.arccos(np.dot(normed_craters_center_point[0], normed_craters_center_point[2])))
    thetas[2] = np.rad2deg(np.arccos(np.dot(normed_craters_center_point[1], normed_craters_center_point[2])))

    return thetas


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

def vote_and_identify(objects_votes):
    # Dictionary to store votes for each object
    votes_dict = {}

    # Process each vote
    for obj, vote in objects_votes:
        if obj not in votes_dict:
            # Initialize voting record for new object
            votes_dict[obj] = {}
        if vote not in votes_dict[obj]:
            votes_dict[obj][vote] = 0
        # Increment the vote count
        votes_dict[obj][vote] += 1

    # Determine the majority class for each object
    majority_class = {}
    for obj, votes in votes_dict.items():
        majority_class[obj] = max(votes, key=votes.get)

    return majority_class


def rc_compute(matched_ids, gt_ids):
    # Convert arrays to sets for efficient operations
    # Create dictionaries to map ids to their positions
    # matched_ids_pos = {id: idx for idx, id in enumerate(matched_ids)}
    # Initialize counts
    TP = FP = FN = TN = 0

    # Compute TP, FP, FN, and TN
    for m_id, gt_id in zip(matched_ids, gt_ids):
        if m_id != 'None' and gt_id != 'None':
            if m_id == gt_id:
                TP += 1
            else:
                FP += 1
        elif m_id == 'None' and gt_id != 'None':
            FN += 1
        elif m_id != 'None' and gt_id == 'None':
            FP += 1
        elif m_id == 'None' and gt_id == 'None':
            TN += 1

    # Compute rates
    TPR = TP / (TP + FN) if TP + FN > 0 else 0
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    FNR = FN / (TP + FN) if TP + FN > 0 else np.nan
    TNR = TN / (FP + TN) if FP + TN > 0 else np.nan

    matching_rate = TP / len([gt_id for gt_id in gt_ids if gt_id != 'None'])
    false_alarm = FP / len(gt_ids)

    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0

    F1 = 2 * (recall * precision) / (recall + precision + 1e-8)

    return TPR, FPR, FN, recall, precision, matching_rate, F1, false_alarm

def draw_ellipse(id, params, loc):
    path = f'data/CH5-png/{loc}.png'
    image = cv2.imread(path)

    pickle_data = []

    for i in range(len(params)):

        if id[i] == 'None':
            continue
        
        center_coordinates = (params[i][0], params[i][1]) 
        axesLength = (params[i][2] * 2, params[i][3] * 2)

        angle = np.rad2deg(params[i][4])

        # Red color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 5 px  
        thickness = 3

        # Using cv2.ellipse() method 
        # Draw a ellipse with red line borders of thickness of 5 px 
        image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)

        image = cv2.putText(image, str(id[i]), (int(params[i][0]), int(params[i][1] - params[i][3] - 20)), 2, 1, 125)

        pickle_data.append((id[i],params[i]))

    #save image
    cv2.imwrite(f'output/images/{loc}.png', image)

    #save correspondance of ellipse parameter to ID
    with open(f'output/pkl/{loc}.pkl', 'wb') as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(f'christian_cid_method-main/output/csv/{loc}.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Ellipse Parameter', 'ID'])

    #     for i in range(len(params)):
    #         writer.writerow([params[i],id[i]])


def process(curr_img_params, image_name, i):
        # cam = camera_extrinsic[i]

    # get gt cam and gt_pos
    # gt_pos = -cam[0:3, 0:3].T @ cam[0:3, 3]
    # noisy_att = noisy_cam_orientations[i]
    # gt_att = noisy_att.T
    # gt_ids = craters_indices[i]
    # curr_noise_lvl = noise_levels[i]

    print(f"Processing frame {i}")

    # Try higher?
    num_sam = 260 # This is a hyperparameter. I sample only 20 since the combinations grow exponentially. This can be increased to trade off time with performance. - max for Ce5 is 250 (doing 260 in case)
    num_sampled = np.minimum(num_sam, len(curr_img_params))
    cc_idx = np.arange(len(curr_img_params))
    if len(curr_img_params) > num_sam:
        random.shuffle(cc_idx)
        cc_idx = cc_idx[:num_sam]

    matched_bool = np.zeros(num_sampled)

    # curr_craters_id = np.array(craters_indices[i][cc_idx])
    CC_params = np.zeros([num_sampled, 5])
    CC_conics = np.zeros([num_sampled, 3, 3])
    sigma_sqr = np.zeros([num_sampled])
    matched_idx = np.zeros([num_sampled])
    matched_ids = [['None'] for _ in range(num_sampled)]
    ncp_match_flag = False
    cp_match_flag = False

    # Convert curr_img_params to CC_conics
    for j in range(len(cc_idx)):
        param = curr_img_params[cc_idx[j]]
        CC_params[j] = param
        CC_conics[j] = ellipse_to_conic_matrix(*param)
        

    found_flag = True
    cc_comb = permutations(range(len(CC_params)), 3)
    count = 0
    object_votes = []
    for comb_id in cc_comb:  # switch
        comb_id = np.array(comb_id)
        curr_comb_CC_conics = CC_conics[comb_id, :]
        curr_comb_CC_params = CC_params[comb_id, :]
        curr_sigma_sqr = sigma_sqr[comb_id]
        # curr_gt_indices = curr_craters_id[comb_id]
        
        descriptors = []

        try:
            CP_des = coplanar_triad_descriptors(curr_comb_CC_conics[0],
                                                        curr_comb_CC_conics[1],
                                                        curr_comb_CC_conics[2])
            
        except ValueError:
            CP_des = None
            continue

        if CP_des is not None:
            distances, indexes = CP_tree.query(CP_des, k=top_n)
            
        craters_id = CP_c_ids[indexes]
        
        for k in range(3):
            object_votes.append(("Imaged_crater_id_" + str(comb_id[k]), craters_id[k]))
        
        count = count + 1
    
    
    #I think this identifies the individual craters from the triads
    result = vote_and_identify(object_votes)
    
    #For each conic param (20 or sample size), put ids into array 
    for k in range(len(CC_params)):
        if 'Imaged_crater_id_' + str(k) in result:
            matched_ids[k] = result['Imaged_crater_id_' + str(k)]
            
        else:
            matched_ids[k] = 'None'

    matched_ids_str = [x.decode('utf-8') for x in matched_ids]

    draw_ellipse(matched_ids, CC_params, image_name)
    

    # If ground-truth IDs are known:
    # TPR, FPR, FN, recall, precision, matching_rate, F1, false_alarm = rc_compute(matched_ids_str, curr_craters_id)

    # result_str = "Testing ID: {} | Matched IDs: {} | Matching_rate: {:.2f}\n".format(
        # str(i), ', '.join(map(str, matched_ids_str)), matching_rate)
    
    # Else:
    result_str = "Testing ID: {} | Matched_IDs: {}\n".format(image_name, ', '.join(matched_ids_str))
    print(result_str)

    return result_str

    # with open(crater_ids_result_dir, 'a') as f:
    #     f.write(result_str)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Script to process data.")
    # parser.add_argument("--dir", required=True, help="Directory path")
    # parser.add_argument("--img_sigma", type=float, required=True, help="img sigma")
    # parser.add_argument("--top_n", type=int, required=True, help="top n")
    # parser.add_argument("--nside", type=int, required=True, help="nside")
    # args = parser.parse_args()
    #
    # dir = args.dir
    # img_sigma = args.img_sigma
    # top_n = args.top_n
    # nside = args.nside

    dir = '.' # set your home dir for this folder

    img_sigma = 3 # found to be the best threshold
    top_n = 1
    local_nside = '32'

    data_dir = 'data'
    output_dir = 'output'
    calibration_file =  data_dir + '/calibration.pkl'

    K = get_intrinsic(calibration_file)

    img_w = 2352
    img_h = 1728
    
    des_dir = data_dir + '/descriptor_db/'
    type = 'local'

    prefix = type + '_coplanar_' + str(local_nside) + "_2K"
    # Download from https://www.dropbox.com/scl/fi/toxsq4vvz25ossfmi8b9p/local_coplanar_32.npz?rlkey=a7nn0g221wjsl5p10anqq6nip&st=6pzgorcc&dl=0
    if os.path.exists(des_dir + f'{prefix}.npz'):
        data = np.load(des_dir + f'{prefix}.npz')
        CP_c_ids = data['c_ids']
        K_values = data['values']
    print('Number of combinations: ' + str(CP_c_ids.shape[0]))

    CP_tree = cKDTree(K_values)

    ## Read the craters database in raw form -- Not USED APPARENTLY! (START)
    # local_craters_database_text_dir = data_dir + '/robbins_navigation_dataset_christians_local.txt'
    # local_CW_params, local_CW_conic, local_CW_conic_inv, local_CW_ENU, local_CW_Hmi_k, local_ID, local_crater_center_point_tree = \
    #     read_crater_database(local_craters_database_text_dir)

    # all_craters_database_text_dir = data_dir + '/robbins_navigation_dataset_christians_all.txt'
    # all_CW_params, all_CW_conic, all_CW_conic_inv, all_CW_ENU, all_CW_Hmi_k, all_ID, all_crater_center_point_tree = \
    #     read_crater_database(all_craters_database_text_dir)
    ## Not used APPARENTLY (END)

    ### Change this part adapt to input    
    # Not all of them are needed. I think only imaged_params/noisy_imaged_params (x,y,a,b,theta), is needed
    testing_data_dir = data_dir + '/testing_data.csv'
    # camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, craters_indices, \
    # heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations = testing_data_reading_general(
    #     testing_data_dir)
    noisy_imaged_params, image_names = testing_data_read_image_params(testing_data_dir)
    ######


    crater_ids_result_dir = output_dir + '/matched_ids.txt'
    
    random.seed(42)

    #TODO: Convert subset db of ce5 landing site into selenographic coordinates
    #TODO: don't have reprojection coz no camera attitude in dataset
    #TODO: scale ellipse parameters in terms of image pixels
    tiemout = 9999999
    result = Parallel(n_jobs=6,max_nbytes=None,timeout=tiemout)(delayed(process)(noisy_imaged_params[i],image_names[i],i) for i in range(len(noisy_imaged_params)))

    with open('output/matched_ids.txt', 'w') as filehandle:       
        filehandle.writelines(result)
    
    # for i in len(result):
    #     draw_ellipse(result[i][1],result[1][2],result[i][3])

    

