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

def sigma_sqr(param):
    img_sigma = 3
    return ((0.85 / np.sqrt(param[2] * param[3])) * img_sigma) ** 2

def gaussian_angle_distance(Ai_params, Aj_params):
    return gaussian_angle(Ai_params, Aj_params) / sigma_sqr(Ai_params)

def perturb_ellipse_params(params, perturbation_factor=0.02):
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
    perturbed_x_center = x_center + random.uniform(-perturbation_factor, perturbation_factor) * x_center
    perturbed_y_center = y_center + random.uniform(-perturbation_factor, perturbation_factor) * y_center
    perturbed_semi_major = semi_major + random.uniform(-perturbation_factor, perturbation_factor) * semi_major
    perturbed_semi_minor = semi_minor + random.uniform(-perturbation_factor, perturbation_factor) * semi_minor
    perturbed_angle = angle + random.uniform(-perturbation_factor, perturbation_factor) * angle

    return [perturbed_x_center, perturbed_y_center, perturbed_semi_major, perturbed_semi_minor, perturbed_angle]


# Fix visualization to make it clear
param_large = [1444.12186035, 838.32548208, 46.26627632,12.50683455, 0]
reproject_large_close = [1.43367907e+03, 8.35729803e+02, 5.87079166e+01, 1.14121564e+01, 7.69089931e-03]
reproject_large_far = [2000, 500, 5.87079166e+01, 1.14121564e+01, 7.69089931e-03]

param_small = [1461.66643065, 1125.30225029, 4.56290463, 1.97037682, 0]
reproject_small_close = [1470, 1120, 4.5, 2, 0]
reproject_small_far = [200, 4.75785715e+02, 8.08646899e+00, 3.85914433e-02, -1.03509037e+00]

s1 = sigma_sqr(param_large)
g1 = gaussian_angle(param_large, reproject_large_close)
g2 = gaussian_angle(param_large, reproject_large_far)
d1 = g1/s1
d2 = g2/s1

image = "../data/CH5-png/0.png"
image = cv2.imread(image)

image = cv2.ellipse(image, ((param_large[0],param_large[1]), (param_large[2]*2,param_large[3]*2), np.rad2deg(param_large[4])), (0,0,255), 2)
image = cv2.ellipse(image, ((reproject_large_close[0],reproject_large_close[1]), (reproject_large_close[2]*2,reproject_large_close[3]*2), np.rad2deg(reproject_large_close[4])), (0,255,0), 2)
image = cv2.ellipse(image, ((reproject_large_far[0],reproject_large_far[1]), (reproject_large_far[2]*2,reproject_large_far[3]*2), np.rad2deg(reproject_large_far[4])), (0,255,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_close[0]), int(reproject_large_close[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far[0]), int(reproject_large_far[1])), (255,0,0), 2)
image = cv2.putText(image, f"SS: {s1}", (int(param_large[0]-190), int(param_large[1]+60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
image = cv2.putText(image, f"GA: {g1}, GA/SS: {d1}", (int(reproject_large_close[0]-200), int(reproject_large_close[1]+100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
image = cv2.putText(image, f"GA: {g2}, GA/SS: {d2}", (int(reproject_large_far[0]-600), int(reproject_large_far[1]-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

s2 = sigma_sqr(param_small)
g3 = gaussian_angle(param_small, reproject_small_close)
g4 = gaussian_angle(param_small, reproject_small_far)
d3 = g3/s2
d4 = g4/s2

image = cv2.ellipse(image, ((param_small[0],param_small[1]), (param_small[2]*2,param_small[3]*2), np.rad2deg(param_small[4])), (0,0,255), 2)
image = cv2.ellipse(image, ((reproject_small_close[0],reproject_small_close[1]), (reproject_small_close[2]*2,reproject_small_close[3]*2), np.rad2deg(reproject_small_close[4])), (0,255,0), 2)
image = cv2.ellipse(image, ((reproject_small_far[0],reproject_small_far[1]), (reproject_small_far[2]*2,reproject_small_far[3]*2), np.rad2deg(reproject_small_far[4])), (0,255,0), 2)
image = cv2.line(image, (int(param_small[0]), int(param_small[1])), (int(reproject_small_close[0]), int(reproject_small_close[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_small[0]), int(param_small[1])), (int(reproject_small_far[0]), int(reproject_small_far[1])), (255,0,0), 2)
image = cv2.putText(image, f"SS: {s2}", (int(param_small[0]-190), int(param_small[1]+60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
image = cv2.putText(image, f"GA: {g3}, GA/SS: {d3}", (int(reproject_small_close[0]-190), int(reproject_small_close[1]+100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
image = cv2.putText(image, f"GA: {g4}, GA/SS: {d4}", (int(reproject_small_far[0]-100), int(reproject_small_far[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
