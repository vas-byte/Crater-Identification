from itertools import combinations, permutations
from descriptor import *
from joblib import Parallel, delayed
from src.get_data import *
from src.utils import *

from numba import njit
from scipy.spatial import cKDTree
import csv
import argparse
import time
import pickle
import cv2

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
        idx = np.where(ID == q.decode('utf-8'))[0]
        if idx.size > 0:
            indices.append(idx[0])
    return np.array(indices)


@njit
def compute_pairwise_ga(remaining_CC_params, neighbouring_craters_id, db_CW_conic, db_CW_Hmi_k, curr_cam, sigma_sqr):
    pairwise_ga = np.ones((remaining_CC_params.shape[0], len(neighbouring_craters_id))) * np.inf
    reproject = np.zeros((remaining_CC_params.shape[0], len(neighbouring_craters_id), 5))

    for ncid in range(len(neighbouring_craters_id)):
        A = conic_from_crater_cpu_mod(db_CW_conic[neighbouring_craters_id[ncid]],
                                      db_CW_Hmi_k[neighbouring_craters_id[ncid]],
                                      curr_cam)  # project them onto the camera
        # convert A to ellipse parameters
        flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)
        # compute ga with all imaged conics
        if np.any(np.isnan(np.array([x_c, y_c, a, b, phi]))):
            continue
        if flag:
            for cc_id in range(remaining_CC_params.shape[0]):
                try:
                    pairwise_ga[cc_id, ncid] = gaussian_angle(remaining_CC_params[cc_id],
                                                              [x_c, y_c, a, b, phi])  # measure pairwise GA
                    reproject[cc_id, ncid] = [x_c, y_c, a, b, phi]
                except:
                    continue

    return pairwise_ga, reproject


def remaining_craters_matching(all_CW_params, all_CW_conic, all_CW_Hmi_k,
                               est_cam, cam_pos, CC_params,
                               CC_conics,
                               all_ID, matched_ids, param_to_id, id_reproject, id_ga, rr_ss, sigma_sqr):
    neighbouring_craters_id = np.arange(all_CW_params.shape[0])

    # 1) project all 3D points onto the image plane
    projected_3D_points = est_cam @ np.hstack(
        [all_CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
    points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                    projected_3D_points[1, :] / projected_3D_points[2, :]])

    within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                        (points_on_img_plane[0, :] <= img_w) &
                                        (points_on_img_plane[1, :] >= 0) &
                                        (points_on_img_plane[1, :] <= img_h) &
                                        ~np.isnan(points_on_img_plane[0, :]) &
                                        ~np.isnan(points_on_img_plane[1, :]))[0]

    fil_ncid = neighbouring_craters_id[within_img_valid_indices]

    # TODO: check if the crater is visible to the camera
    _, fil_ncid = visible_points_on_sphere(all_CW_params[:, 0:3], np.array([0, 0, 0]),
                                           np.linalg.norm(all_CW_params[0, 0:3]),
                                           cam_pos, fil_ncid)
    num_matches = 0

    if not(len(fil_ncid) == 0):
        # Find indices that are not in comb_id
        all_indices = np.arange(CC_conics.shape[0])
        # remaining_indices = np.setdiff1d(all_indices, ordered_comb_id)
        remaining_indices = all_indices

        # Extract the subset from all_craters using the remaining indices
        remaining_CC_params = CC_params[remaining_indices]
        remaining_sigma_sqr = sigma_sqr[remaining_indices]
        # find their NN

        pairwise_ga, reproject = compute_pairwise_ga(remaining_CC_params, fil_ncid, all_CW_conic, all_CW_Hmi_k, est_cam, sigma_sqr)

        # Find the index of the closest neighbouring_craters_id for each remaining_CC_params
        nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(pairwise_ga)
        closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]


        for j in range(len(remaining_indices)):
            if pairwise_ga[j, nearest_neighbors_idx[j]] / remaining_sigma_sqr[j] <= 13.277:
                matched_ids[remaining_indices[j]] = all_ID[closest_neighbouring_ids[j]]

                # get ellipse parameter
                param_to_id[str(all_ID[closest_neighbouring_ids[j]])] = remaining_CC_params[j]
                id_reproject[str(all_ID[closest_neighbouring_ids[j]])] = reproject[j, nearest_neighbors_idx[j]]
                id_ga[str(all_ID[closest_neighbouring_ids[j]])] = pairwise_ga[j, nearest_neighbors_idx[j]]
                rr_ss[str(all_ID[closest_neighbouring_ids[j]])] = remaining_sigma_sqr[j]


                num_matches = num_matches + 1
            else:
                matched_ids[remaining_indices[j]] = 'None'

    return matched_ids, num_matches, param_to_id, id_reproject, id_ga, rr_ss


def angular_distance_rot_mat(R1, R2):
    # Relative rotation matrix
    R_rel = np.dot(R1.T, R2)
    
    # The angular distance is the angle of the relative rotation matrix
    angle_distance = np.arccos((np.trace(R_rel) - 1) / 2)
    
    return angle_distance

def reprojection_test_w_pnp_solver(des_tree, descriptors, c_ids, ID,
                      db_CW_conic, db_CW_Hmi_k, db_CW_params, db_CW_ENU, K, 
                      CC_params, ordered_CC_params, ordered_sigma_sqr,
                      comb_id, matched_ids, top_n, matching_percentage, 
                      param_to_id, id_reproject, id_ga, rr_ss, CC_conics, sigma_sqr):
    
    distances, indexes = des_tree.query(descriptors, k=top_n)

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

        # change this part to use pnp_solver
        # Assume zero distortion for simplicity
        dist_coeffs = np.zeros(4)

        curr_center_3D = np.ascontiguousarray(curr_CW_params[:, 0:3]).reshape((3, 1 ,3))
        curr_center_2D = np.ascontiguousarray(ordered_CC_params[:, 0:2]).reshape((3, 1, 2))
        # Use the P3P solver in OpenCV
        try:
            success, rvecs, tvecs = cv2.solvePnP(
                curr_center_3D, 
                curr_center_2D, 
                K, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_SQPNP
            )
        except:
            continue
        
        if not(success):
            continue
        
        # convert rvecs to rotation matrix
        est_R = cv2.Rodrigues(rvecs)

        est_cam = np.zeros([3, 4])
        est_cam[0:3, 0:3] = est_R[0]
        est_cam[0:3, 3] = tvecs[:, 0]
        est_cam = K @ est_cam
        cam_in_world_coord = - est_R[0].T @ tvecs[:, 0]

        # cam_in_world_coord = NLLS_rm(ordered_CC_params, ordered_CC_conics, curr_CW_params, curr_CW_Conic, gt_att, curr_CW_ENU, K)
        # cam_in_world_coord = np.squeeze(cam_in_world_coord)
        # est_cam = np.zeros([3, 4])
        # est_cam[0:3, 0:3] = gt_att
        # est_cam[0:3, 3] = -gt_att @ cam_in_world_coord
        # # project all craters centers to image plane
        # est_cam = K @ est_cam

        matched = np.zeros(3)

        for j in range(3):
            
            A = conic_from_crater_cpu_mod(curr_CW_Conic[j], curr_CW_Hmi_k[j], est_cam)

            # convert A to ellipse parameters
            flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)

            if np.any(np.isnan([x_c, y_c, a, b, phi])):
                continue

            if (flag):  # if it's proper conic
                try:
                    curr_ga = gaussian_angle(ordered_CC_params[j], [x_c, y_c, a, b, phi])
                except:
                    continue
                if ((curr_ga / ordered_sigma_sqr[j]) <= 13.277):
                    matched[j] = 1


        if np.sum(matched) == 3:
            # redefines matched_ids here
            param_to_id = {}
            id_reproject = {}
            id_ga = {}
            rr_ss = {}
            matched_ids = [[] for _ in range(CC_conics.shape[0])]

            for j in range(3):
                matched_ids[comb_id[j]] = craters_id[j]
            # TODO: do NN match with the remaining craters
            matched_ids, num_matches, param_to_id, id_reproject, id_ga, rr_ss = remaining_craters_matching(db_CW_params, db_CW_conic, db_CW_Hmi_k,
                                                                  est_cam, cam_in_world_coord,
                                                                  CC_params, CC_conics,
                                                                  ID, matched_ids, param_to_id, id_reproject, id_ga, rr_ss,sigma_sqr)
            if ((num_matches) > (matching_percentage * CC_conics.shape[0])):  
                return True, est_cam, cam_in_world_coord, matched_idx, matched_ids, param_to_id, id_reproject, id_ga, rr_ss

    return False, est_cam, cam_in_world_coord, matched_idx, matched_ids, param_to_id, id_reproject, id_ga, rr_ss



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


def save_output(param_to_id, loc):
    output_dir = 'output/'
    img_dir = f'data/CH5-png/{loc}.png'

    
    image = cv2.imread(img_dir)

    pickle_data = []

    for id,param in param_to_id.items():

        pickle_data.append((id,param))

        if id == 'None':
            continue
        
        center_coordinates = (param[0], param[1]) 
        axesLength = (param[2] * 2, param[3] * 2)

        angle = np.rad2deg(param[4])

        # Red color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 5 px  
        thickness = 3

        # Using cv2.ellipse() method 
        # Draw a ellipse with red line borders of thickness of 5 px 
        image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)

        image = cv2.putText(image, str(id), (int(param[0]), int(param[1] - param[3] - 20)), 2, 1, 125)


    #save image
    save_path = output_dir + f'images/{loc}.png'
    cv2.imwrite(save_path, image)

    #save correspondance of ellipse parameter to ID
    save_pkl = output_dir + f'pkl/{loc}.pkl'
    with open(save_pkl, 'wb') as handle:
        pickle.dump(pickle_data, handle)


def log_result(matched_ids, opt_cam_pos, elapsed_time, result_dir, i, param_to_id, img_name):
    
    opt_cam_pos_str = ', '.join(['{:.2f}'.format(val) for val in opt_cam_pos])
    
    # Format the results in a single line
    result_str = ("Testing ID: {} | Matched IDs: {} | Est Pos: {} | Time: {:.2f}\n").format(
        i, ', '.join(str(id) for id in matched_ids), opt_cam_pos_str, elapsed_time)

    save_output(param_to_id,img_name)

    return result_str

def plot_reprojection(params_to_id, id_reproject, id_ga, rr_ss, image_name):
   
    output_dir = 'output/'
    img_dir = f'data/CH5-png/{image_name}.png'

    image = cv2.imread(img_dir)

    print("Params_to_id,id_reproject,id_ga,rr_ss")
    print(len(params_to_id),len(id_reproject),len(id_ga),len(rr_ss))

    for key in params_to_id.keys():
        param = params_to_id[key]
        reproject = id_reproject[key]

        image = cv2.ellipse(image, ((param[0],param[1]), (param[2]*2,param[3]*2), np.rad2deg(param[4])), (0,0,255), 2)
        image = cv2.ellipse(image, ((reproject[0],reproject[1]), (reproject[2]*2,reproject[3]*2), reproject[4]), (0,255,0), 2)
        image = cv2.line(image, (int(param[0]), int(param[1])), (int(reproject[0]), int(reproject[1])), (255,0,0), 2)
        image = cv2.putText(image, str(id_ga[key]/rr_ss[key]), (int(param[0]), int(param[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_dir + 'reprojected/' + f'{image_name}.png', image)

    data = {}

    for key in params_to_id.keys():
        param = params_to_id[key]
        reproject = id_reproject[key]
        rr_ss_val = rr_ss[key]

        data[key] = {'original': param, 'reprojected': reproject, 'sigma': rr_ss_val, 'ga': id_ga[key]}
    
    with open(output_dir + 'reprojected_pkl/' + f'{image_name}.pkl', 'wb') as handle:
        pickle.dump(data, handle)

def crater_area(crater):
    a,b = crater[2],crater[3]
    return math.pi*a*b   

def process(curr_img_params,img_name,i,CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree,CP_tree,CP_c_ids,matching_percentage):
    print('processing ' + str(i))
        
    param_to_id = {}
    id_reproject = {}
    id_ga = {}
    rr_ss = {}

    if len(curr_img_params) < 3:
        result_str = "Testing ID: {} | Not enough imaged craters \n".format(str(i))
        return result_str

    start_time = time.time()  ###################### start time ####################################

    CC_params = np.zeros([len(curr_img_params), 5])
    CC_conics = np.zeros([len(curr_img_params), 3, 3])
    sigma_sqr = np.zeros([len(curr_img_params)])
    matched_idx = np.zeros([len(curr_img_params)])
    matched_ids = [[] for _ in range(len(curr_img_params))]
    ncp_match_flag = False
    cp_match_flag = False

    # Convert curr_img_params to CC_conics and compute sigma_sqr
    for j, param in enumerate(curr_img_params):
        CC_params[j] = param
        CC_conics[j] = ellipse_to_conic_matrix(*param)
        sigma_sqr[j] = ((0.85 / np.sqrt(param[2] * param[3])) * img_sigma) ** 2

    # Compute invariants
    cc_comb = permutations(range(len(CC_params)), 3)
    found_flag = True

    for comb_id in cc_comb:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print("Time limit exceeded, breaking the loop")
            break
        comb_id = np.array(comb_id)
        curr_comb_CC_conics = CC_conics[comb_id, :]
        curr_comb_CC_params = CC_params[comb_id, :]
        curr_sigma_sqr = sigma_sqr[comb_id]
        
        try:
            CP_des = coplanar_triad_descriptors(curr_comb_CC_conics[0],
                                                curr_comb_CC_conics[1],
                                                curr_comb_CC_conics[2])
        except ValueError:
            CP_des = None

        if CP_des is not None and not np.any(np.isnan(CP_des)):
            if solver == 'pnp':
                cp_match_flag, est_cam, cam_pos, matched_idx, matched_ids, param_to_id, id_reproject, id_ga, rr_ss = reprojection_test_w_pnp_solver(CP_tree, CP_des,
                                                                                                CP_c_ids,
                                                                                                ID,
                                                                                                CW_conic,
                                                                                                CW_Hmi_k,
                                                                                                CW_params,
                                                                                                CW_ENU, K,
                                                                                                CC_params, curr_comb_CC_params,
                                                                                                curr_sigma_sqr,
                                                                                                comb_id,
                                                                                                matched_ids,
                                                                                                top_n, 
                                                                                                matching_percentage, 
                                                                                                param_to_id, 
                                                                                                id_reproject, 
                                                                                                id_ga, 
                                                                                                rr_ss, 
                                                                                                CC_conics, 
                                                                                                sigma_sqr)

        if cp_match_flag:
            break

    ############ Important logic 1 ####################
    ####### if no match at all for all combinations, skip the rest #############
    if not (ncp_match_flag) and not (cp_match_flag):
        # if both no match at all, go to the result saving
        for j in range(len(matched_ids)):
            matched_ids[j] = 'None'

        cam_pos_str = ', '.join(['{:.2f}'.format(val) for val in cam_pos])
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        result_str = ("Testing ID: {} | Matched IDs: {} | Est Pos: {} | Time: {:.2f}\n").format(
            i, ', '.join(str(id) for id in matched_ids), cam_pos_str, elapsed_time)

        return result_str


    end_time = time.time()
    elapsed_time = end_time - start_time

    plot_reprojection(param_to_id, id_reproject, id_ga, rr_ss, img_name)
    return log_result(matched_ids, cam_pos, elapsed_time, result_dir, i, param_to_id, img_name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Script to process data.")
    # parser.add_argument("--dir", required=True, help="Directory path")
    # parser.add_argument("--testing_data_dir", required=True, help="Directory path")
    # parser.add_argument("--img_sigma", type=float, required=True, help="img sigma")
    # parser.add_argument("--result_dir", required=True, help="Output path")
    # parser.add_argument("--starting_id", type=int, required=True, help="noise_lvl")
    # parser.add_argument("--step", type=int, required=True, help="noise_lvl")
    # parser.add_argument("--local_nside", type=int, required=True, help="local_nside")
    # parser.add_argument("--global_nside", type=int, required=True, help="global_nside")
    # parser.add_argument("--top_n", type=int, required=True, help="top_n")
    # parser.add_argument("--matching_percentage", type=float, required=True, help="top_n")
    # parser.add_argument("--time_limit", type=int, required=True, help="time_limit")
    # args = parser.parse_args()

    # dir = args.dir
    # img_sigma = args.img_sigma
    # local_nside = args.local_nside
    # global_nside = args.global_nside
    # testing_data_dir = args.testing_data_dir
    # top_n = args.top_n
    # result_dir = args.result_dir
    # time_limit = args.time_limit
    # starting_id = args.starting_id
    # matching_percentage = args.matching_percentage
    # step = args.step
    # ending_id = args.starting_id + step

    # dir = '/data/Dropbox/craters/christian_craters_ID/'
    # # dir = '/media/ckchng/1TBHDD/Dropbox/craters/christian_craters_ID/'
    # img_sigma = 3
    # nside = 32
    # top_n = 10

    dir = ''
    test_data = 'nadir_pointing'
    testing_data_dir = 'data/testing_data.csv'
    img_sigma = 3
    local_nside = 32

    top_n = 10
    time_limit = 1800 # 30 mins
    starting_id = 0
    matching_percentage = 0.5
    step = 500
    # ending_id = starting_id + step
    solver = 'pnp'
    result_dir = 'output/' + str(starting_id) + '_' + solver + '_'+ test_data+'.txt'
    
    data_dir = dir + 'data/'
    output_dir = dir + 'output/'
    calibration_file = data_dir + 'calibration.pkl'
    K = get_intrinsic(calibration_file)

    img_w = 2352
    img_h = 1728

    des_dir = data_dir + 'descriptor_db/'
    ###################### Read local features ######################################3
    # CP_c_ids, K_values = read_descriptors(des_dir)
    #TODO: replace this with the descriptor database that you built
    prefix = 'local_coplanar_' + str(local_nside) + "_2K"
    data = np.load(des_dir + prefix + '.npz')
    CP_c_ids = data['c_ids']
    K_values = data['values']

    # prefix = 'local_coplanar_' + str(nside)
    # CP_c_ids, K_values = descriptor_reading_pkl(des_dir, prefix, 'local')
    CP_tree = cKDTree(K_values)
    print('Number of combinations: ' + str(CP_c_ids.shape[0]))

    ### Read the craters database in raw form
    #TODO: replace this with the catalogue subset that you extracted
    craters_database_text_dir = data_dir + 'descriptor_db/filtered_catalog_2K.txt' 
    CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree = \
        read_crater_database(craters_database_text_dir)

    noisy_imaged_params, image_names = testing_data_read_image_params(testing_data_dir)

    result = Parallel(n_jobs=6,max_nbytes=None)(delayed(process)(noisy_imaged_params[i],image_names[i],i,CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree,CP_tree,CP_c_ids,matching_percentage) for i in range(len(noisy_imaged_params)))

    with open(result_dir, 'w') as filehandle:       
        filehandle.writelines(result)
    
    
