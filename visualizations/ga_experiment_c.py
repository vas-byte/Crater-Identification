import numpy as np
import random
import cv2

def calc_c(Ai_params, Aj_params):
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

    return diff.T @ Yi @ Y_inv @ Yj @ diff


# Fix visualization to make it clear
param_large = [1444.12186035, 838.32548208, 46.26627632,12.50683455, 0]
reproject_large_close = [1.45367907e+03, 8.38729803e+02, 5.87079166e+01, 1.14121564e+01, 7.69089931e-03]
reproject_large_far = [2000, 500, 5.87079166e+01, 1.14121564e+01, 7.69089931e-03]
reproject_large_far_two = [2000, 500, 5.87079166e+01, 1.14121564e+01, 3]
reproject_large_far_three = [2000, 500, 8.87079166e+01, 1.14121564e+01, 7.69089931e-03]
reproject_large_far_four = [2000, 500, 5.87079166e+01, 4.14121564e+01, 7.69089931e-03]
reproject_large_far_five = [1700, 600, 5.87079166e+01, 4.14121564e+01, 7.69089931e-03]

g1 = calc_c(param_large, reproject_large_close)
g2 = calc_c(param_large, reproject_large_far)
g3 = calc_c(param_large, reproject_large_far_two)
g4 = calc_c(param_large, reproject_large_far_three)
g5 = calc_c(param_large, reproject_large_far_four)
g6 = calc_c(param_large, reproject_large_far_five)

image = "../data/CH5-png/0.png"
image = cv2.imread(image)

image = cv2.ellipse(image, ((param_large[0],param_large[1]), (param_large[2]*2,param_large[3]*2), np.rad2deg(param_large[4])), (0,0,255), 2)
image = cv2.ellipse(image, ((reproject_large_close[0],reproject_large_close[1]), (reproject_large_close[2]*2,reproject_large_close[3]*2), np.rad2deg(reproject_large_close[4])), (0,255,0), 2)
image = cv2.ellipse(image, ((reproject_large_far[0],reproject_large_far[1]), (reproject_large_far[2]*2,reproject_large_far[3]*2), np.rad2deg(reproject_large_far[4])), (50, 205, 50), 2)
image = cv2.ellipse(image, ((reproject_large_far_two[0],reproject_large_far_two[1]), (reproject_large_far_two[2]*2,reproject_large_far_two[3]*2), np.rad2deg(reproject_large_far_two[4])), (50, 205, 154), 2)
image = cv2.ellipse(image, ((reproject_large_far_three[0],reproject_large_far_three[1]), (reproject_large_far_three[2]*2,reproject_large_far_three[3]*2), np.rad2deg(reproject_large_far_three[4])), (0, 255, 255), 2)
image = cv2.ellipse(image, ((reproject_large_far_four[0],reproject_large_far_four[1]), (reproject_large_far_four[2]*2,reproject_large_far_four[3]*2), np.rad2deg(reproject_large_far_four[4])), (255,255,0), 2)
image = cv2.ellipse(image, ((reproject_large_far_five[0],reproject_large_far_five[1]), (reproject_large_far_five[2]*2,reproject_large_far_five[3]*2), np.rad2deg(reproject_large_far_five[4])), (255,255,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_close[0]), int(reproject_large_close[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far[0]), int(reproject_large_far[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far_two[0]), int(reproject_large_far_two[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far_three[0]), int(reproject_large_far_three[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far_four[0]), int(reproject_large_far_four[1])), (255,0,0), 2)
image = cv2.line(image, (int(param_large[0]), int(param_large[1])), (int(reproject_large_far_five[0]), int(reproject_large_far_five[1])), (173, 255, 47), 2)
image = cv2.putText(image, f"c: {g1}", (int(reproject_large_close[0]-200), int(reproject_large_close[1]+100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
image = cv2.putText(image, f"c: {g2}", (int(reproject_large_far[0]-200), int(reproject_large_far[1]-200)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)
image = cv2.putText(image, f"c: {g3}", (int(reproject_large_far_two[0]-200), int(reproject_large_far_two[1]-150)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 154), 2)
image = cv2.putText(image, f"c: {g4}", (int(reproject_large_far_three[0]-200), int(reproject_large_far_three[1]-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
image = cv2.putText(image, f"c: {g5}", (int(reproject_large_far_four[0]-200), int(reproject_large_far_four[1]-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
image = cv2.putText(image, f"c: {g6}", (int(reproject_large_far_five[0]-200), int(reproject_large_far_five[1]-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (173, 255, 47), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
