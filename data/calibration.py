import numpy as np
import pickle

fx = fy = 15.4/0.0074

# Prepare
w, h = 2352, 1728

# Go
fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

print("Field of View (degrees):")
print(f"  {fov_x =}\N{DEGREE SIGN}")
print(f"  {fov_y =}\N{DEGREE SIGN}")

cx = 1164.01684
cy = 858.04100

K = np.array([[fov_x, 0, cx],
              [0, fov_y, cy],
              [0, 0, 1]])

with open('calibration.pkl', 'wb') as f:
    pickle.dump(K, f)

print("Calibration:")
print(K)

