import numpy as np
import math
import numba
from numba import njit
import os
os.environ['OPENCV_IO_ENABLE_JASPER']='true'
import cv2
import os
import matplotlib.pyplot as plt


@njit
def custom_meshgrid(x, y, z):  # tested
    nx, ny, nz = len(x), len(y), len(z)

    x_grid = np.empty((ny, nx, nz))
    y_grid = np.empty((ny, nx, nz))
    z_grid = np.empty((ny, nx, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_grid[j, i, k] = x[i]
                y_grid[j, i, k] = y[j]
                z_grid[j, i, k] = z[k]

    return x_grid, y_grid, z_grid

def custom_meshgrid_2d(x, y):  # tested
    nx, ny= len(x), len(y)

    x_grid = np.empty((ny, nx))
    y_grid = np.empty((ny, nx))

    for i in range(nx):
        for j in range(ny):
            x_grid[j, i] = x[i]
            y_grid[j, i] = y[j]


    return x_grid, y_grid





def add_noise_to_craters_cam(craters_cam, a_noise, b_noise, x_noise, y_noise, phi_noise):
    noisy_craters_cam = np.zeros([craters_cam.shape[0], 3, 3])
    noisy_craters_params = np.zeros([craters_cam.shape[0], 5])
    a_noise_pct = []
    b_noise_pct = []
    x_noise_pct = []
    y_noise_pct = []
    for i in range(len(craters_cam)):
        curr_x = craters_cam[i, 0] + x_noise[i]
        curr_y = craters_cam[i, 1] + y_noise[i]
        curr_a = craters_cam[i, 2] + a_noise[i]
        curr_b = craters_cam[i, 3] + b_noise[i]
        curr_phi = craters_cam[i, 4] + phi_noise[i]
        x_noise_pct.append(x_noise[i] / craters_cam[i, 0])
        y_noise_pct.append(y_noise[i] / craters_cam[i, 1])
        a_noise_pct.append(a_noise[i] / craters_cam[i, 2])
        b_noise_pct.append(b_noise[i] / craters_cam[i, 3])

        noisy_craters_params[i] = curr_x, curr_y, curr_a, curr_b, curr_phi
        noisy_craters_cam[i] = ellipse_to_conic_matrix(curr_x, curr_y, curr_a, curr_b, curr_phi)

    return noisy_craters_cam, noisy_craters_params, x_noise_pct, y_noise_pct, a_noise_pct, b_noise_pct

def craters_weight_computation(craters_params):
    size = []
    for i in range(len(craters_params)):
        curr_a = craters_params[i, 2]
        curr_b = craters_params[i, 3]
        size.append(curr_a * curr_b)

    sum = np.sum(np.array(size))
    weights = np.array(size) / sum

    return weights


def generate_oriented_bbox_points_cpu(CC_params, num_sam):
    N = CC_params.shape[0]
    MAX_SAMPLES = num_sam ** 2  # Set this to the maximum number of samples you expect

    rotated_points_x = np.zeros((N, MAX_SAMPLES))
    rotated_points_y = np.zeros((N, MAX_SAMPLES))
    level_curve_a = np.zeros((N, MAX_SAMPLES))

    for k in range(N):
        xc, yc, a, b, phi = CC_params[k]

        x_samples = np.linspace(-a, a, num_sam)
        y_samples = np.linspace(-b, b, num_sam)

        x, y = np.meshgrid(x_samples, y_samples)

        R = np.array([[math.cos(phi), -math.sin(phi)],
                      [math.sin(phi), math.cos(phi)]])

        r = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
        D_a = R @ r @ np.transpose(R)

        idx = 0
        for i in range(num_sam):
            for j in range(num_sam):
                point = np.array([x[i, j], y[i, j]])
                rotated_point = np.dot(R, point)
                rotated_points_x[k, idx] = rotated_point[0] + xc
                rotated_points_y[k, idx] = rotated_point[1] + yc

                disp_a = np.array([rotated_points_x[k, idx], rotated_points_y[k, idx]]) - np.array([xc, yc])
                level_curve_a[k, idx] = np.transpose(disp_a) @ D_a @ disp_a
                idx += 1

    return rotated_points_x, rotated_points_y, level_curve_a

# # Example usage:
# CC_params = np.array([[0, 0, 1, 1, 0], [1, 1, 2, 2, math.pi / 4]])
# rotated_points_x, rotated_points_y = generate_oriented_bbox_points_cpu(CC_params, 10)



def round_up(value, decimals=2):
    return math.floor(value * 10**decimals) / 10**decimals

def angular_distance(R1, R2):
    """
    Compute the angular distance between two rotation matrices R1 and R2.

    Parameters:
    - R1, R2: Rotation matrices.

    Returns:
    - Angular distance in radians.
    """
    # Compute the relative rotation matrix
    # R = np.dot(R2, R1.T)
    R = np.dot(R1.T, R2)
    # Compute the angle of rotation
    theta = np.arccos((np.trace(R) - 1) / 2.0)

    return np.rad2deg(theta)


# @njit
# def ellipse_centre_euclid_distance(A, C):
#     _, x_a, y_a, _, _, _ = extract_ellipse_parameters_from_conic(A)
#     _, x_c, y_c, _, _, _ = extract_ellipse_parameters_from_conic(C)
#     euclid_dist = np.linalg.norm(np.array([x_a, y_a]) - np.array([x_c, y_c]))
#     return euclid_dist

def compute_optimal_pose_cpu(sampled_position, sampled_attitude, K, CW_conic_inv, Hmi_k, CC_conic, CC_params,
                             min_sum_obj_vals, min_sum_obj_ids):
    '''
    :param sampled_position: [I x 3]
    :param sampled_attitude: [J x 3 x 3]
    :param K: [3 x 3]
    :param CW_conic_inv: [M x 3 x 3]
    :param Hmi_k: [M x 4 x 3]
    :param CC_conic: [M x 3 x 3]
    :param min_sum_obj_vals: [I]
    :param min_sum_obj_ids: [I]
    :return:
    '''
    n_positions = sampled_position.shape[0]
    n_attitudes = sampled_attitude.shape[0]

    for i in range(n_positions):
        neg_curr_position = sampled_position[i, :]
        sum_obj_val = np.zeros(n_attitudes)

        # Initialize so3 matrix
        so3 = np.zeros((3, 4))

        inner_obj_val = np.zeros(CW_conic_inv.shape[0])

        for j in range(n_attitudes):
            curr_att = sampled_attitude[j]
            rc = np.dot(curr_att, neg_curr_position)

            for row in range(3):
                for col in range(3):
                    so3[row, col] = curr_att[row, col]
                so3[row, 3] = rc[row]

            P_mc = np.dot(K, so3)

            for k in range(CW_conic_inv.shape[0]):
                curr_A = CC_conic[k]
                curr_C = conic_from_crater_cpu(CW_conic_inv[k], Hmi_k[k], P_mc)

                inner_obj_val[k] = ellipse_centre_euclid_distance(curr_A, curr_C)

            sum_obj_val[j] = np.sum(inner_obj_val)

        min_sum_obj_ids[i] = np.argmin(sum_obj_val)
        min_sum_obj_vals[i] = sum_obj_val[int(min_sum_obj_ids[i])]

    overall_min_idx = int(np.argmin(min_sum_obj_vals))
    obj_val = min_sum_obj_vals[int(overall_min_idx)]
    opt_pos = sampled_position[int(overall_min_idx), :]
    opt_att = sampled_attitude[int(min_sum_obj_ids[overall_min_idx])]

    return opt_pos, opt_att, obj_val


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

# Get a conic matrix from an ellipse.
@njit
def ellipse_to_conic_matrix(x, y, a, b, phi):
    A = a**2*((np.sin(phi))**2)+b**2*((np.cos(phi))**2)
    B = 2*(b**2-a**2)*np.cos(phi)*np.sin(phi)
    C = a**2*((np.cos(phi))**2)+b**2*((np.sin(phi))**2)
    D = -2*A*x-B*y
    E = -B*x-2*C*y
    F = A*x**2+B*x*y+C*y**2-a**2*b**2

    # TODO: do i need to normalise here?

    return np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])



def save_contour_and_append_info(filename, textfile, pos_id, att_id, max_pos_id, max_att_id, min_obj_vals, opt_pos_vals, opt_att_vals, pos_res,
                                 att_res, gt_pos, gt_att, write_text=True):
    """
    Save contour plots to a file and append relevant information to a text file.

    Parameters:
    - filename: Name of the file to save the figure.
    - textfile: Name of the text file to append the information.
    - pos_id, att_id: 1D arrays for the axes of the contour plots.
    - min_obj_vals: List of min_obj_val matrices (e.g., [min_obj_val_eu, min_obj_val_ga, min_obj_val_ls])
    - opt_pos_vals: List of opt_pos values (e.g., [opt_pos_eu, opt_pos_ga, opt_pos_ls])
    - opt_att_vals: List of opt_att values (e.g., [opt_att_eu, opt_att_ga, opt_att_ls])
    - pos_res: Position resolution.
    - att_res: Attitude resolution.
    - angular_distance: Helper function to compute angular distance between two rotation matrices.
    """

    titles = ['Contour for min_obj_val_eu', 'Contour for min_obj_val_ga', 'Contour for min_obj_val_ls', 'Contour for min_obj_val_el']
    labels = ['PnP', 'PnE_GA', 'PnE_Level', 'PnE_ED']

    # Generate meshgrid for pos_id and att_id
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    with open(textfile, 'a') as f:
        for i in range(4):
            row = i // 2  # Determine the row index
            col = i % 2  # Determine the column index

            # Compute Euclidean and angular differences
            eu_diff = np.linalg.norm(-opt_pos_vals[i].reshape([1, 3]) - gt_pos)
            pos_id_val = int(eu_diff / pos_res) * pos_res
            angular_diff = angular_distance(opt_att_vals[i], gt_att)

            att_id_val = int(angular_diff / att_res) * att_res

            curr_max_pos_id = int(np.maximum((pos_id_val / pos_res) + 2, max_pos_id))
            curr_max_att_id = int(np.maximum((att_id_val / att_res) + 2, max_att_id))
            X, Y = np.meshgrid(pos_id[0:curr_max_pos_id], att_id[0:curr_max_att_id])
            # Contour plot
            contour = ax[row, col].contourf(X, Y, min_obj_vals[i][0:curr_max_att_id, 0:curr_max_pos_id], 25)
            ax[row, col].set_title(titles[i])
            ax[row, col].set_xlabel('pos_id')
            ax[row, col].set_ylabel('att_id')

            # Set x and y ticks based on the determined intervals
            ax[row, col].set_xticks(np.arange(min(pos_id), max(pos_id) + pos_res, pos_res))
            ax[row, col].set_yticks(np.arange(min(att_id), max(att_id) + att_res, att_res))

            # Scatter plot with red cross annotation
            ax[row, col].scatter(pos_id_val, att_id_val, color='red', marker='x')

            fig.colorbar(contour, ax=ax[row, col])

            if (write_text):
                # Append information to the text file
                f.write(f"Results for {labels[i]}:\n")
                f.write(f"opt_pos_{labels[i]}: {opt_pos_vals[i]}\n")
                f.write(f"opt_att_{labels[i]}: {opt_att_vals[i]}\n")
                f.write(f"min_obj_val_{labels[i]}: {np.min(min_obj_vals[i])}\n")
                f.write(f"eu_diff_{labels[i]}: {eu_diff}\n")
                f.write(f"angular_diff_{labels[i]}: {angular_diff}\n\n")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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
    # if np.min(np.array(non_visible_len_P_cam)) < np.max(np.array(visible_len_P_cam)):
    #     print('Something is wrong\n')

    return visible_points, visible_indices

def create_extrinsic_matrix(plane_normal, radius, rotate=False):
    # Ensure the plane normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Camera's z-axis is the opposite of the plane normal
    z_axis = -plane_normal

    # Determine an up vector. If the z-axis is not parallel to [0, 1, 0], use [0, 1, 0] as the up vector.
    # Otherwise, use [1, 0, 0].
    if np.abs(np.dot(z_axis, [0, 1, 0])) != 1:
        up_vector = [0, 1, 0]
    else:
        up_vector = [1, 0, 0]

    # Camera's x-axis
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Camera's y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix
    R = np.array([x_axis, y_axis, z_axis]).T

    rotation_angle = 0
    if rotate:
        # Compute a random rotation angle between 0 and 60 degrees
        rotation_angle = np.random.uniform(0, np.radians(60))

        # Create a rotation matrix around a random axis
        random_axis = np.random.rand(3)
        random_axis = random_axis / np.linalg.norm(random_axis)

        rand_rot_mat = axis_angle_to_rotation_matrix_scipy(random_axis, rotation_angle)
        # Apply the random rotation to R
        R = R @ rand_rot_mat

    # Translation vector (camera's position in world coordinates)
    t = plane_normal * radius

    # Extrinsic matrix
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t  # Convert world position to camera-centric position
    # extrinsic[3, 3] = 1

    return extrinsic, np.degrees(rotation_angle)

from scipy.spatial.transform import Rotation
def axis_angle_to_rotation_matrix_scipy(axis, angle):
    """
    Convert axis-angle to rotation matrix using scipy.

    Parameters:
    - axis: A 3D unit vector representing the rotation axis.
    - angle: Rotation angle in radians.

    Returns:
    - 3x3 rotation matrix.
    """
    r = Rotation.from_rotvec(axis * angle)
    return r.as_matrix()

def conic_matrix_to_ellipse(cm):
    A = cm[0][0]
    B = cm[0][1] * 2
    C = cm[1][1]
    D = cm[0][2] * 2
    E = cm[1][2] * 2
    F = cm[2][2]

    x_c = (2 * C * D - B * E) / (B ** 2 - 4 * A * C)
    y_c = (2 * A * E - B * D) / (B ** 2 - 4 * A * C)

    if ((B ** 2 - 4 * A * C) >= 0):
        return 0, 0, 0, 0, 0

    try:
        a = math.sqrt((2 * (A * E ** 2 + C * D ** 2 - B * D * E + F * (B ** 2 - 4 * A * C))) / (
                    (B ** 2 - 4 * A * C) * (math.sqrt((A - C) ** 2 + B ** 2) - A - C)))
        b = math.sqrt((2 * (A * E ** 2 + C * D ** 2 - B * D * E + F * (B ** 2 - 4 * A * C))) / (
                    (B ** 2 - 4 * A * C) * (-1 * math.sqrt((A - C) ** 2 + B ** 2) - A - C)))

        phi = 0
        if (B == 0 and A > C):
            phi = math.pi / 2
        elif (B != 0 and A <= C):
            phi = 0.5 * math.acot((A - C) / B)
        elif (B != 0 and A > C):
            phi = math.pi / 2 + 0.5 * math.acot((A - C) / B)

        return x_c, y_c, a, b, phi

    except:
        return 0, 0, 0, 0, 0

@njit
def extract_ellipse_parameters_from_conic(conic):
    A = conic[0, 0]
    B = conic[0, 1] * 2
    C = conic[1, 1]
    D = conic[0, 2] * 2
    F = conic[1, 2] * 2
    G = conic[2, 2]

    # Sanity test.
    denominator = B ** 2 - 4 * A * C
    if (B ** 2 - 4 * A * C >= 0) or (C * np.linalg.det(conic) >= 0):
        # print('Conic equation is not a nondegenerate ellipse')
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

    #  Method from:
    #  https://en.wikipedia.org/wiki/Ellipse
    #  Convention in wikipedia:
    #   [ A B/2  D/2]
    #   [ B/2 C  E/2]
    #   [ D/2 E/2 F]]
    #  The following equations reexpresses wikipedia's formulae in Christian et
    #  al.'s convention.

    # Get centres.
    try:
        x_c = (2 * C * D - B * F) / denominator
        y_c = (2 * A * F - B * D) / denominator

        # Get semimajor and semiminor axes.
        KK = 2 * (A * F ** 2 + C * D ** 2 - B * D * F + (B ** 2 - 4 * A * C) * G)
        root = math.sqrt((A - C) ** 2 + B ** 2)
        a = -1 * math.sqrt(KK * ((A + C) + root)) / denominator
        b = -1 * math.sqrt(KK * ((A + C) - root)) / denominator

        if B != 0:
            phi = math.atan((C - A - root) / B)  # Wikipedia had this as acot; should be atan. Check https://math.stackexchange.com/questions/1839510/how-to-get-the-correct-angle-of-the-ellipse-after-approximation/1840050#1840050
        elif A < C:
            phi = 0
        else:
            phi = math.pi / 2

        return True, x_c, y_c, a, b, phi
    except:
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001


def draw_ellipses_on_image_ls(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k, get_image, conic_from_crater_cpu,
                           extract_ellipse_parameters_from_conic, opt_pos_vals, opt_att_vals, out_img_path):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # magenta, blue, and black
    color_scheme = [(0, 0, 0)]

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=5)

        # then measurements
        curr_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            curr_color = color_scheme[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_param = extract_ellipse_parameters_from_conic(projected_conic)
            center_coordinates = (int(curr_param[1]), int(curr_param[2]))
            axesLength = (int(curr_param[3]), int(curr_param[4]))
            angle = int(curr_param[5] * 180 / math.pi)
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, curr_color, 2)
            image = cv2.circle(image, center_coordinates, radius=0, color=curr_color, thickness=5)

    # Add legend to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'black = PnE_Level', (10, 160), font, 1, color_scheme[0], 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image


def draw_ellipses_on_image_ed(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k, get_image, conic_from_crater_cpu,
                           extract_ellipse_parameters_from_conic, opt_pos_vals, opt_att_vals, out_img_path):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # magenta, blue, and black
    color_scheme = [(255, 0, 255)]

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=5)

        # then measurements
        curr_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            curr_color = color_scheme[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_param = extract_ellipse_parameters_from_conic(projected_conic)
            center_coordinates = (int(curr_param[1]), int(curr_param[2]))
            axesLength = (int(curr_param[3]), int(curr_param[4]))
            angle = int(curr_param[5] * 180 / math.pi)
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, curr_color, 2)
            image = cv2.circle(image, center_coordinates, radius=0, color=curr_color, thickness=5)

    # Add legend to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'black = PnE_ED', (10, 190), font, 1, color_scheme[0], 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image

def draw_ellipses_on_image(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k, get_image, conic_from_crater_cpu,
                           extract_ellipse_parameters_from_conic, opt_pos_vals, opt_att_vals, out_img_path):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # magenta, blue, and black
    color_scheme = [(0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 0, 255)]

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=5)

        # then measurements
        curr_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            curr_color = color_scheme[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_param = extract_ellipse_parameters_from_conic(projected_conic)
            center_coordinates = (int(curr_param[1]), int(curr_param[2]))
            axesLength = (int(curr_param[3]), int(curr_param[4]))
            angle = int(curr_param[5] * 180 / math.pi)
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, curr_color, 2)
            image = cv2.circle(image, center_coordinates, radius=0, color=curr_color, thickness=5)

    # Add legend to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'yellow = PnP', (10, 100), font, 1, color_scheme[0], 2, cv2.LINE_AA)
    cv2.putText(image, 'blue = PnE_GA', (10, 130), font, 1, color_scheme[1], 2, cv2.LINE_AA)
    cv2.putText(image, 'black = PnE_Level', (10, 160), font, 1, color_scheme[2], 2, cv2.LINE_AA)
    cv2.putText(image, 'magenta = PnE_ED', (10, 190), font, 1, color_scheme[3], 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image

def draw_ellipses_on_image_pnp_ed(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k, get_image, conic_from_crater_cpu,
                            opt_pos_vals, opt_att_vals, out_img_path):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # magenta, blue, and black
    color_scheme = [(0, 255, 255), (255, 0, 255)]

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=8)

        # then measurements
        curr_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            curr_color = color_scheme[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_param = extract_ellipse_parameters_from_conic(projected_conic)
            center_coordinates = (int(curr_param[1]), int(curr_param[2]))
            axesLength = (int(curr_param[3]), int(curr_param[4]))
            angle = int(curr_param[5] * 180 / math.pi)
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, curr_color, 2)
            image = cv2.circle(image, center_coordinates, radius=0, color=curr_color, thickness=5)

    # Add legend to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'yellow = PnP', (10, 100), font, 1, color_scheme[0], 2, cv2.LINE_AA)
    cv2.putText(image, 'magenta = PnE_ED', (10, 190), font, 1, color_scheme[1], 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image

def ellipse_distance_vis(gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k,
                                         conic_from_crater_cpu,
                                     opt_pos_vals, opt_att_vals):
    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    gt_xy_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])
    gt_ab_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])
    gt_phi_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])

    mea_xy_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])
    mea_ab_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])
    mea_phi_dev = np.zeros([CW_conic_inv.shape[0], len(opt_pos_vals)])
    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_CW_param = extract_ellipse_parameters_from_conic(projected_conic)
        gt_center_coordinates = np.array([curr_CW_param[1], curr_CW_param[2]])
        gt_axesLength = np.array([curr_CW_param[3], curr_CW_param[4]])
        gt_angle = curr_CW_param[5]

        # then measurements
        curr_CC_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        mea_center_coordinates = np.array([curr_CC_param[1], curr_CC_param[2]])
        mea_axesLength = np.array([curr_CC_param[3], curr_CC_param[4]])
        mea_angle = curr_CC_param[5]

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_CW_param = extract_ellipse_parameters_from_conic(projected_conic)
            est_center_coordinates = np.array([curr_CW_param[1], curr_CW_param[2]])
            est_axesLength = np.array([curr_CW_param[3], curr_CW_param[4]])
            est_angle = curr_CW_param[5]

            gt_xy_dev[i, j] = np.linalg.norm(est_center_coordinates - gt_center_coordinates)
            gt_ab_dev[i, j] = np.linalg.norm(est_axesLength - gt_axesLength)
            gt_phi_dev[i, j] = np.rad2deg(np.abs(est_angle - gt_angle))

            mea_xy_dev[i, j] = np.linalg.norm(est_center_coordinates - mea_center_coordinates)
            mea_ab_dev[i, j] = np.linalg.norm(est_axesLength - mea_axesLength)
            mea_phi_dev[i, j] = np.rad2deg(np.abs(est_angle - mea_angle))


    mean_gt_xy_dev = np.mean(gt_xy_dev, axis=0)
    mean_gt_ab_dev = np.mean(gt_ab_dev, axis=0)
    mean_gt_phi_dev = np.mean(gt_phi_dev, axis=0)

    mean_mea_xy_dev = np.mean(mea_xy_dev, axis=0)
    mean_mea_ab_dev = np.mean(mea_ab_dev, axis=0)
    mean_mea_phi_dev = np.mean(mea_phi_dev, axis=0)

    return mean_gt_xy_dev, mean_gt_ab_dev, mean_gt_phi_dev, mean_mea_xy_dev, mean_mea_ab_dev, mean_mea_phi_dev






def draw_ellipses_and_write_res_on_image(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k,
                                         get_image, conic_from_crater_cpu,
                                     opt_pos_vals, opt_att_vals, out_img_path,
                                         loss_funcs, rot_x, rot_y, level_curve_a):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # magenta, blue, and black
    color_scheme = [(0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 0, 255)]
    offset = np.array([[10, 0], [10, 30], [10, 60], [10, 90]])
    #

    # get diff P_mc
    opt_P_mcs = []
    for i in range(len(opt_pos_vals)):
        curr_opt_pos = opt_pos_vals[i]
        curr_opt_att = opt_att_vals[i]
        rc = np.dot(curr_opt_att, curr_opt_pos)
        so3 = np.zeros([3, 4])
        so3[0:3, 0:3] = curr_opt_att
        so3[:, 3] = rc
        opt_P_mcs.append(np.dot(K, so3))

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_CW_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_CW_param[1]), int(curr_CW_param[2]))
        axesLength = (int(curr_CW_param[3]), int(curr_CW_param[4]))
        angle = int(curr_CW_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=5)

        # then measurements
        curr_CC_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_CC_param[1]), int(curr_CC_param[2]))
        axesLength = (int(curr_CC_param[3]), int(curr_CC_param[4]))
        angle = int(curr_CC_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

        for j in range(len(opt_pos_vals)):
            curr_P_mc = opt_P_mcs[j]
            curr_color = color_scheme[j]
            projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], curr_P_mc)
            curr_CW_param = extract_ellipse_parameters_from_conic(projected_conic)

            curr_func = loss_funcs[j]
            if (j == 0) or (j == 1): # eu
                curr_res = curr_func(CC_conic[i], projected_conic)
            elif j == 2:
                curr_res = curr_func(curr_CW_param, rot_x[i],
                                       rot_y[i], level_curve_a[i])
            elif j == 3:
                curr_res = curr_func(curr_CC_param, curr_CW_param)


            cv2.putText(image, f'{curr_res:.2f}', center_coordinates + offset[j], font, 1, color_scheme[j], 2, cv2.LINE_AA)

            center_coordinates = (int(curr_CW_param[1]), int(curr_CW_param[2]))
            axesLength = (int(curr_CW_param[3]), int(curr_CW_param[4]))
            angle = int(curr_CW_param[5] * 180 / math.pi)
            image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, curr_color, 2)
            image = cv2.circle(image, center_coordinates, radius=0, color=curr_color, thickness=5)



    # Add legend to the image

    # cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(image, 'yellow = PnP', (10, 100), font, 1, color_scheme[0], 2, cv2.LINE_AA)
    # cv2.putText(image, 'blue = PnE_GA', (10, 130), font, 1, color_scheme[1], 2, cv2.LINE_AA)
    # cv2.putText(image, 'black = PnE_Level', (10, 160), font, 1, color_scheme[2], 2, cv2.LINE_AA)
    # cv2.putText(image, 'magenta = PnE_ED', (10, 190), font, 1, color_scheme[3], 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image


def draw_ellipses_on_image_with_mea_and_gt(image_path, gt_att, gt_pos, K, CC_conic, CW_conic_inv, CW_Hmi_k, get_image, conic_from_crater_cpu,
                           extract_ellipse_parameters_from_conic, out_img_path):
    """
    Draw ellipses on the image based on provided parameters and functions.

    Parameters:
    - image_path: Path to the image.
    - gt_att, gt_pos, K, CW_conic_inv, CW_Hmi_k: Parameters for ellipse computation.
    - get_image: Function to read the image.
    - conic_from_crater_cpu: Function to compute the projected conic.
    - extract_ellipse_parameters_from_conic: Function to extract ellipse parameters from the conic.

    Returns:
    - image: Image with drawn ellipses.
    """

    # Read the image
    image = get_image(os.path.abspath(image_path))

    # Compute rc, so3, and P_mc
    rc = np.dot(gt_att, -gt_pos)
    so3 = np.zeros([3, 4])
    so3[0:3, 0:3] = gt_att
    so3[:, 3] = rc
    P_mc = np.dot(K, so3)

    # Get params and draw ellipses
    for i in range(CW_conic_inv.shape[0]):
        # first gt
        projected_conic = conic_from_crater_cpu(CW_conic_inv[i], CW_Hmi_k[i], P_mc)
        curr_param = extract_ellipse_parameters_from_conic(projected_conic)
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 255, 0), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 255, 0), thickness=5)

        # then measurements
        curr_param = extract_ellipse_parameters_from_conic(CC_conic[i])
        center_coordinates = (int(curr_param[1]), int(curr_param[2]))
        axesLength = (int(curr_param[3]), int(curr_param[4]))
        angle = int(curr_param[5] * 180 / math.pi)
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0, 0, 255), 2)
        image = cv2.circle(image, center_coordinates, radius=0, color=(0, 0, 255), thickness=5)

    # Add legend to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'red = mea', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'green = GT', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, image)
    return image

@njit
def conic_from_crater_cpu(C_conic_inv, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # print('yo')
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci, C_conic_inv, 3, 3, 3)
    Astar = matrix_multiply_cpu(Astar, Hci.T, 3, 3, 3)
    # A = np.linalg.inv(Astar)
    legit_flag, A = inverse_3x3_cpu(Astar)

    return A


def differentiate_values(vec):
    # Compute pairwise absolute differences
    diffs = np.abs([
        vec[0] - vec[1],
        vec[0] - vec[2],
        vec[1] - vec[2]
    ])

    # Find the indices of the two smallest differences
    sorted_indices = np.argsort(diffs)

    # Use the indices to determine the repeated and unique values
    if sorted_indices[0] == 0:  # vec[0] and vec[1] are close
        same_value = np.mean([vec[0], vec[1]])
        unique_value = vec[2]
        unique_idx = 2
    elif sorted_indices[0] == 1:  # vec[0] and vec[2] are close
        same_value = np.mean([vec[0], vec[2]])
        unique_value = vec[1]
        unique_idx = 1
    else:  # vec[1] and vec[2] are close
        same_value = np.mean([vec[1], vec[2]])
        unique_value = vec[0]
        unique_idx = 0

    return same_value, unique_value, unique_idx


@njit
def differentiate_values_numba(vec):
    # Compute pairwise absolute differences using statically-sized arrays
    diffs = np.empty(3)
    diffs[0] = np.abs(vec[0] - vec[1])
    diffs[1] = np.abs(vec[0] - vec[2])
    diffs[2] = np.abs(vec[1] - vec[2])

    # Find the indices of the two smallest differences
    sorted_indices = np.argsort(diffs)

    # Use the indices to determine the repeated and unique values
    if sorted_indices[0] == 0:  # vec[0] and vec[1] are close
        same_value = (vec[0] + vec[1]) / 2
        unique_value = vec[2]
        unique_idx = 2
    elif sorted_indices[0] == 1:  # vec[0] and vec[2] are close
        same_value = (vec[0] + vec[2]) / 2
        unique_value = vec[1]
        unique_idx = 1
    else:  # vec[1] and vec[2] are close
        same_value = (vec[1] + vec[2]) / 2
        unique_value = vec[0]
        unique_idx = 0

    return same_value, unique_value, unique_idx

@njit
def conic_from_crater_cpu_mod(C_conic, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    Hci_inv = np.linalg.inv(Hci)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci_inv.T, C_conic, 3, 3, 3)
    A = matrix_multiply_cpu(Astar, Hci_inv, 3, 3, 3)

    # A_ = Hci.T @ A @ Hci
    # A = np.linalg.inv(Astar)
    # A = inverse_3x3_cpu(Astar)
    return A

def imaged_conic_to_crater_conic(C_conic, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply_cpu(Pm_c, Hmi_k, 3, 4, 3)
    # Hci_inv = np.linalg.inv(Hci)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply_cpu(Hci.T, C_conic, 3, 3, 3)
    A = matrix_multiply_cpu(Astar, Hci, 3, 3, 3)
    # A = np.linalg.inv(Astar)
    # A = inverse_3x3_cpu(Astar)
    return A


@njit
def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])



@njit
def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@njit
def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

@njit
def intrinsic_zxz_rotation(alpha, beta, gamma):
    zx_rot = rotation_matrix_x(beta) @ (rotation_matrix_z(gamma))
    return rotation_matrix_z(alpha) @ (zx_rot)



@njit
def extrinsic_xyz_rotation(alpha, beta, gamma):
    return rotation_matrix_x(alpha) @ (rotation_matrix_y(beta) @ rotation_matrix_z(gamma))


@njit
def rotation_compute(yaw_rad, pitch_rad, roll_rad):
    R_w_ci_intrinsic = intrinsic_zxz_rotation(0.0, -np.pi / 2, 0.0)
    R_ci_cf_intrinsic = intrinsic_zxz_rotation(yaw_rad, pitch_rad, 0.0)
    R_c_intrinsic = R_ci_cf_intrinsic @ R_w_ci_intrinsic
    R_w_c_extrinsic = np.transpose(R_c_intrinsic)
    R_c_roll_extrinsic = extrinsic_xyz_rotation(0.0, 0.0, roll_rad)
    R = R_c_roll_extrinsic @ R_w_c_extrinsic
    return R

@njit
def matrix_multiply_cpu(A, B, A_rows, A_cols, B_cols):
    C = np.zeros((A_rows, B_cols))
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C

@njit
def inverse_3x3_cpu(A):

    detA = A[0, 0] * (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + \
           A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])

    invA = np.zeros_like(A)

    try:
        invDetA = 1.0 / detA
        invA[0, 0] = (A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]) * invDetA
        invA[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) * invDetA
        invA[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) * invDetA
        invA[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) * invDetA
        invA[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) * invDetA
        invA[1, 2] = (A[1, 0] * A[0, 2] - A[0, 0] * A[1, 2]) * invDetA
        invA[2, 0] = (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1]) * invDetA
        invA[2, 1] = (A[2, 0] * A[0, 1] - A[0, 0] * A[2, 1]) * invDetA
        invA[2, 2] = (A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]) * invDetA
        legit_flag = True
    except:
        legit_flag = False
    return legit_flag, invA


# def oriented_bounding_box(x_c, y_c, a, b, theta):
#     # Bounding box corners
#     top_left = (x_c - a * math.cos(theta), y_c - a * math.sin(theta))
#     top_right = (x_c + a * math.cos(theta), y_c + a * math.sin(theta))
#     bottom_left = (x_c - b * math.sin(theta), y_c + b * math.cos(theta))
#     bottom_right = (x_c + b * math.sin(theta), y_c - b * math.cos(theta))
#
#     # Bounding box parameters
#     obb_center = (x_c, y_c)
#     obb_width = 2 * a
#     obb_height = 2 * b
#     obb_orientation = theta
#
#     return obb_center, obb_width, obb_height, obb_orientation


def oriented_bounding_box(x_c, y_c, a, b, theta):
    # Bounding box corners
    # top_left = (x_c - a * math.cos(theta), y_c - b * math.sin(theta))
    # top_right = (x_c + a * math.cos(theta), y_c - b * math.sin(theta))
    # bottom_left = (x_c - a * math.sin(theta), y_c + b * math.cos(theta))
    # bottom_right = (x_c + a * math.sin(theta), y_c + b * math.cos(theta))

    top_left = np.array([-a, -b])
    top_right = np.array([a, -b])
    btm_left = np.array([-a, b])
    btm_right = np.array([a, b])

    R = np.array([[math.cos(theta), -1 * math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])

    rotated_tl = R @ top_left + np.array([x_c, y_c])
    rotated_tr = R @ top_right + np.array([x_c, y_c])
    rotated_bl = R @ btm_left + np.array([x_c, y_c])
    rotated_br = R @ btm_right + np.array([x_c, y_c])

    # Bounding box parameters
    obb_center = (x_c, y_c)
    obb_width = 2 * a
    obb_height = 2 * b
    obb_orientation = theta

    return rotated_tl, rotated_tr, rotated_bl, rotated_br