import numpy as np
import random
import cv2

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

def perturb_ellipse_params(params, perturbation_factor=0.005):
    """
    Perturb the given ellipse parameters slightly.
    
    Parameters:
    - params: list of ellipse parameters in the form [x_center, y_center, semi_major, semi_minor, angle]
    - perturbation_factor: float, maximum percentage of perturbation for each parameter
    
    Returns:
    - perturbed_params: list of perturbed ellipse parameters
    """
    x_center, y_center, semi_major, semi_minor, angle = params

    # Perturb each parameter by a small random value, up to perturbation_factor percentage of the original value
    perturbed_x_center = x_center + random.uniform(-perturbation_factor, perturbation_factor) * 2352
    perturbed_y_center = y_center + random.uniform(-perturbation_factor, perturbation_factor) * 1728
    perturbed_semi_major = semi_major + random.uniform(-perturbation_factor, perturbation_factor) * semi_major
    perturbed_semi_minor = semi_minor + random.uniform(-perturbation_factor, perturbation_factor) * semi_minor
    perturbed_angle = angle + random.uniform(-perturbation_factor, perturbation_factor) * np.radians(10)  # Perturb angle within a small range

    return [perturbed_x_center, perturbed_y_center, perturbed_semi_major, perturbed_semi_minor, perturbed_angle]


param = [10.516640363109582*2352/100,41.557557024159394*1728/100,1.2257156600360826*2352/100,0.3670396382452734*1728/100,np.radians(355.314100160498)]
param2 = [5.11564091394203*2352/100,90.527386482902*1728/100,0.17611606364520682*2352/100,0.11985869004966077*1728/100,np.radians(0)]

image = "../data/CH5-png/0.png"
image = cv2.imread(image)

ga1 = 0
ga2 = 0

for i in range(10):
    pert = perturb_ellipse_params(param,0.009)
    ga1 = max(gaussian_angle(param, pert),ga1)
    image = cv2.ellipse(image, ((pert[0],pert[1]), (pert[2]*2,pert[3]*2), np.rad2deg(pert[4])), (0,0,255), 1)

for i in range(10):
    pert = perturb_ellipse_params(param2,0.003)
    image = cv2.ellipse(image, ((pert[0],pert[1]), (pert[2]*2,pert[3]*2), np.rad2deg(pert[4])), (0,0,255), 1)
    ga2 = max(gaussian_angle(param2, pert),ga2)

image = cv2.ellipse(image, ((param[0],param[1]), (param[2]*2,param[3]*2), np.rad2deg(param[4])), (0,255,0), 2)
image = cv2.ellipse(image, ((param2[0],param2[1]), (param2[2]*2,param2[3]*2), np.rad2deg(param2[4])), (0,255,0), 2)

print(ga1, ga2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()