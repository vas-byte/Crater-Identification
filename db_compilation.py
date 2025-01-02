from itertools import combinations

from src.get_data import *
from src.utils import *

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

    Yi_len = np.array([[1/a_i**2, 0], [0, 1/ b_i **2]])
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
    exp_part  = np.exp(-0.5 * diff.T @ Yi @ Y_inv @ Yj @ diff)

    front_part = (4 * np.sqrt(Yi_det * Yj_det)) / np.linalg.det(Y_sum)

    dGA = np.arccos(np.minimum(front_part * exp_part, 1))
    return dGA**2

def strip_symbols(s, symbols):
    for symbol in symbols:
        s = s.replace(symbol, '')
    return s

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
#
#
# def find_full_match_indices(query, ID):
#     """
#     Finds the indices of rows in the 2D array ID where each row contains all elements of the 1D array query,
#     regardless of the order of elements.
#
#     Parameters:
#     - query: 1D array of strings, representing the query elements.
#     - ID: 2D array of strings, representing the array to be searched.
#
#     Returns:
#     - A numpy array of indices of the rows in ID that fully match the query.
#     """
#     indices = []
#     query_set = set(query)  # Convert the query to a set for efficient comparison
#     query_len = len(query)
#
#     for idx, row in enumerate(ID):
#         row_set = set(row)
#         if row_set == query_set and len(row) == query_len:
#             return idx
#             # indices.append(idx)
#     return indices


@njit
def compute_pairwise_ga(remaining_CC_params, neighbouring_craters_id, db_CW_conic, db_CW_Hmi_k, curr_cam):
    pairwise_ga = np.ones((remaining_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    for ncid in range(len(neighbouring_craters_id)):
        A = conic_from_crater_cpu_mod(db_CW_conic[neighbouring_craters_id[ncid]], db_CW_Hmi_k[neighbouring_craters_id[ncid]],
                                      curr_cam)  # project them onto the camera
        # convert A to ellipse parameters
        flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)
        # compute ga with all imaged conics
        if np.any(np.isnan(np.array([x_c, y_c, a, b, phi]))):
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
    values = []   # List to store the remaining values of each line

    with open(filename, 'r') as file:
        for line in file:
            try:
                curr_values = line.strip().split()
                ids.append(curr_values[:3])     # First 3 values as strings
                values.append([float(x) for x in curr_values[3:]])  # Remaining values as floats
            except:
                continue

    # Convert lists to Numpy arrays
    ids = np.array(ids)
    values = np.array(values)

    return ids, values

def read_crater_database(craters_database_text_dir, to_be_removed_dir):
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
    with open(to_be_removed_dir, 'r') as file:
        removed_ids = [line.strip() for line in file.readlines()]

    # Find the indices of these IDs in the ID array
    removed_indices = np.where(np.isin(ID, removed_ids))[0]

    # Remove the craters with indices in removed_indices from your data arrays
    db_CW_params = np.delete(db_CW_params, removed_indices, axis=0)
    db_CW_conic = np.delete(db_CW_conic, removed_indices, axis=0)
    db_CW_conic_inv = np.delete(db_CW_conic_inv, removed_indices, axis=0)
    db_CW_ENU = np.delete(db_CW_ENU, removed_indices, axis=0)
    db_CW_Hmi_k = np.delete(db_CW_Hmi_k, removed_indices, axis=0)
    ID = np.delete(ID, removed_indices, axis=0)

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

def descriptor_reading_pkl(main_dir, prefix, type, N):
    # Iterate over all files in the directory
    max_cid_length = 11
    
    # Initialize arrays
    cid = np.empty((N, 3), dtype=f'S{max_cid_length}')
    if type == 'local':
        descriptors = np.empty((N, 7))
    elif type == 'global':
        descriptors = np.empty((N, 3))
    current_row = 0
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

                # here, dont use list, use a predefined matrix.
                # Determine the number of rows to update
                num_rows = len(curr_cid)
                
                # Update the arrays
                try:
                    cid[current_row:current_row + num_rows, :] = np.array(curr_cid, dtype=f'S{max_cid_length}')
                    descriptors[current_row:current_row + num_rows, :] = curr_descriptors

                    # Update the row index
                    current_row += num_rows
                    
                except:
                    print("Dimension Error: ", num_rows, curr_cid, curr_descriptors)

    # cid = np.vstack(cid)
    # descriptors = np.vstack(descriptors)

    return cid, descriptors

# def descriptor_reading_pkl(main_dir, prefix, type):
#     # Initialize HDF5 file
#     with h5py.File(main_dir + prefix + '.h5', 'w') as hf:
#         # Estimate maximum length of combined cid strings
#         max_cid_length = 40  # Adjust as needed

#         # String dtype for flattened cids
#         str_dtype = h5py.string_dtype('utf-8', max_cid_length)

#         # Initial size of datasets
#         initial_size = 2900000
#         descriptor_length = 7  # Adjust based on the length of your descriptor tuples

#         # Create resizable datasets
#         cids_dset = hf.create_dataset('cids', shape=(initial_size,), maxshape=(None,), dtype=str_dtype)
#         descriptors_dset = hf.create_dataset('descriptors', shape=(initial_size, descriptor_length), maxshape=(None, descriptor_length), dtype='f')

#         row_c = 0
#         row_d = 0

#         for filename in os.listdir(main_dir):
#             if filename.startswith(prefix):
#                 filepath = os.path.join(main_dir, filename)

#                 with open(filepath, 'rb') as file:
#                     loaded_array = pickle.load(file)
#                     # ... Your processing logic ...

#                     curr_cid, curr_descriptors = lists_iterating(loaded_array, type)

#                     # Flatten cids tuples and convert to byte strings
#                     flattened_cids = [';'.join(cid_tuple).encode('utf-8') for cid_tuple in curr_cid]
#                     fixed_length_cids = np.array(flattened_cids, dtype=str_dtype)

#                     # Flatten cids tuples and convert to fixed-length numpy strings
                    
#                     # flattened_cids = [';'.join(cid_tuple) for cid_tuple in curr_cid]
#                     # fixed_length_cids = np.array(flattened_cids, dtype='S{}'.format(max_cid_length))

#                     # Resize datasets if needed
#                     if row_c + len(curr_cid) > cids_dset.shape[0]:
#                         cids_dset.resize((row_c + len(curr_cid),))
#                     if row_d + len(curr_descriptors) > descriptors_dset.shape[0]:
#                         descriptors_dset.resize((row_d + len(curr_descriptors), descriptor_length))

#                     # Storing the data
#                     cids_dset[row_c:row_c + len(curr_cid)] = fixed_length_cids
#                     descriptors_dset[row_d:row_d + len(curr_descriptors), :] = curr_descriptors

#                     # Update row counters
#                     row_c += len(curr_cid)
#                     row_d += len(curr_descriptors)

import sys
def combination_and_mem_counting(main_dir, prefix, type, text_dir):
    # Iterate over all files in the directory
    cid = []
    descriptors = []
    total_memory = 0
    total_combinations = 0
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

                # Calculate lengths
                length_cid = len(curr_cid)
                # length_descriptors = len(curr_descriptors)

                # Calculate sizes in bytes
                size_cid = sys.getsizeof(np.array(curr_cid))
                size_descriptors = sys.getsizeof(np.array(curr_descriptors))

                # Update total combinations and memory
                total_combinations += length_cid
                total_memory += size_cid + size_descriptors

                with open(text_dir, 'a') as f:
                    f.write(filename + ' ' + str(total_combinations) + ' ' +  str(round(total_memory / (1024 ** 2), 2)) + '\n')

                #
                # cid.append(np.array(curr_cid))
                # descriptors.append(np.array(curr_descriptors))

    # cid = np.vstack(cid)
    # descriptors = np.vstack(descriptors)

    return total_combinations, total_memory



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Script to process data.")
    # parser.add_argument("--dir", required=True, help="Directory path")
    # parser.add_argument("--text_dir", required=True, help="Directory path")
    # parser.add_argument("--out_dir", required=True, help="Directory path")
    # parser.add_argument("--nside", type=int, required=True, help="nside")
    # parser.add_argument("--db_type", required=True, help="db_type")
    
    # args = parser.parse_args()
    # #
    # dir = args.dir
    # text_dir = args.text_dir
    # out_dir = args.out_dir
    # nside = args.nside
    # db_type = args.db_type

    dir = 'data/descriptor_db/pkl'
    text_dir = 'data/descriptor_db/db_compilation.txt'
    out_dir = 'data/descriptor_db/'
    db_type = 'local'
    nside = 32
    
    
    des_dir = dir
    
    if db_type == 'local':
        prefix = db_type + '_coplanar_' + str(nside)
    elif db_type == 'global':
        prefix = db_type + '_non_coplanar_' + str(nside)

    total_combo, total_memory = combination_and_mem_counting(des_dir, prefix, db_type, text_dir)
    print(total_combo)
    print(total_memory)
    # Using h5py to save the data
    
    c_ids, values = descriptor_reading_pkl(des_dir, prefix, db_type, total_combo)
    np.savez(out_dir + prefix + '.npz', c_ids=c_ids, values=values)
  


