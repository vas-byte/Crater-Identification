import pickle
import cv2
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

def find_indices(query, ID):
    indices = []
    for q in query:
        idx = np.where(ID == q)[0]
        if idx.size > 0:
            indices.append(idx[0])
    return np.array(indices)

def reprojection_test_w_pnp_solver(ID,
                      db_CW_conic, db_CW_Hmi_k, db_CW_params, db_CW_ENU, K, 
                      ordered_CC_params, craters_id):
    

    # pick N best descriptors
    est_cam = np.zeros([3, 4])
    rm = np.zeros([3])

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

    success, rvecs, tvecs = cv2.solvePnP(
        curr_center_3D, 
        curr_center_2D, 
        K, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_SQPNP
    )
 
    
    # convert rvecs to rotation matrix
    est_R = cv2.Rodrigues(rvecs)

    est_cam = np.zeros([3, 4])
    est_cam[0:3, 0:3] = est_R[0]
    est_cam[0:3, 3] = tvecs[:, 0]
    est_cam = K @ est_cam
    cam_in_world_coord = - est_R[0].T @ tvecs[:, 0]

    return est_cam, cam_in_world_coord

@njit
def get_craters_world_numba(lines):
    # Initialize the matrices
    N = len(lines)
    crater_param = np.zeros((N, 6))
    crater_conic = np.zeros((N, 3, 3))
    crater_conic_inv = np.zeros((N, 3, 3))
    Hmi_k = np.zeros((N, 4, 3))
    ENU = np.zeros((N, 3, 3))
    S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    # Populate the matrices
    k = np.array([0, 0, 1])

    for idx, line in enumerate(lines):
        X, Y, Z, a, b, phi = line
        a = (a * 1000) / 2 # converting diameter to meter
        b = (b * 1000) / 2
        phi = phi

        # Populate crater_param
        crater_param[idx] = [X, Y, Z, a, b, phi]

        # Calculate conic matrix
        A = a ** 2 * (np.sin(phi) ** 2) + b ** 2 * (np.cos(phi) ** 2)
        B = 2 * (b ** 2 - a ** 2) * np.cos(phi) * np.sin(phi)
        C = a ** 2 * (np.cos(phi) ** 2) + b ** 2 * (np.sin(phi) ** 2)
        D = -2 * A * 0 - B * 0
        E = -B * 0 - 2 * C * 0
        F = A * 0 ** 2 + B * 0 * 0 + C * 0 ** 2 - a ** 2 * b ** 2

        # Populate crater_conic
        # crater_conic[idx] = [[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]]
        crater_conic[idx] = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])

        crater_conic_inv[idx] = np.linalg.inv(crater_conic[idx])

        # get ENU coordinate
        Pc_M = np.array([X, Y, Z])

        u = Pc_M / np.linalg.norm(Pc_M)
        e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

        TE_M = np.empty((3, 3), dtype=np.float64)
        TE_M[:, 0] = e
        TE_M[:, 1] = n
        TE_M[:, 2] = u

        ENU[idx] = TE_M
        # compute Hmi

        Hmi = np.hstack((TE_M.dot(S), Pc_M.reshape(-1, 1)))
        Hmi_k[idx] = np.vstack((Hmi,  k.reshape(1, 3)))

    return crater_param, crater_conic, crater_conic_inv, ENU, Hmi_k

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

def get_intrinsic(calibration_file):
    
    with open(calibration_file, "rb") as f:
        intrinsic = pickle.load(f)
    
    return intrinsic

def arrayLongestPattern(array):
    deviation = 1

    ids = set(array)
    max_count = 0
    max_id = "None"
    ids.remove("None")

    for i in ids:
        count = 0
        for j in range(0, len(array)):
            if array[j] == i:
                count += 1

                if deviation < 1:
                    deviation += 1

            else:
                if deviation > 0:
                    deviation -= 1
                    count += 1
                else:

                    if count > max_count:
                        max_count = count
                        max_id = i

                    deviation = 1
                    count = 0
    
    return max_id, max_count

# Build dictionary
height = 1728
width = 2352

data = {}

successive_path = "../data/Successive Craters/params.pickle"

with open(successive_path, 'rb') as handle:
    b = pickle.load(handle)

SuccessiveCount = {}

for i in range(0,1337):
    SuccessiveCount[i] = 0

for i in range(0,101):
    for crater in b[i]:
        SuccessiveCount[crater[0]] += 1

for i in range(110,420,10):
    for crater in b[i]:
        SuccessiveCount[crater[0]] += 1


for i in range(0,1337):
    data[str(i)] = []

 
for i in range(0,101):

    path =  f"../output/pkl/{i}.pkl"
    successive_ids = []

    try:    
        with open(path, mode ='rb') as file:    
        
            params_data = pickle.load(file)                            

            for param_data in params_data:
                
                    param = param_data[1]
                    cid = param_data[0]
                    successive_id = -1

                    for crater in b[i]:
                    
                        p1 = crater[1][0] * width/100
                        p2 = crater[1][1] * height/100
                        p3 = crater[1][2] * width/100
                        p4 = crater[1][3] * height/100
                        p5 = np.radians(crater[1][4])

                        param_list = np.array([p1,p2,p3,p4,p5])

                        if (param_list == param).all():
                            successive_id = crater[0]
                            break

                    if successive_id == -1:
                        continue

                    if SuccessiveCount[successive_id] < 10:
                        continue
                    
                
                    if str(successive_id) in data:
                        successive_ids.append(successive_id)
                        data[str(successive_id)].append(str(cid))
                    else:
                        data[str(successive_id)] = [str(cid)]
            
            for key in list(data.keys()):
                if int(key) not in successive_ids:
                    if key in data:
                        data[key].append("None")
                    else:
                        data[key] = ["None"]

    except Exception as e:
        print(e)
        print("Skipped ",i)
        continue

for i in range(110,420,10):

    path = f"../output/pkl/{i}.pkl"
    successive_ids = []

    try:
        with open(path, mode ='rb') as file:    
        
            params_data = pickle.load(file)

            for param_data in params_data:
                
                    param = param_data[1]
                    cid = param_data[0]
                    successive_id = -1

                    for crater in b[i]:
                    
                        p1 = crater[1][0] * width/100
                        p2 = crater[1][1] * height/100
                        p3 = crater[1][2] * width/100
                        p4 = crater[1][3] * height/100
                        p5 = np.radians(crater[1][4])

                        param_list = np.array([p1,p2,p3,p4,p5])

                        if (param_list == param).all():
                            successive_id = crater[0]
                            break

                    if successive_id == -1:
                        continue

                    if SuccessiveCount[successive_id] < 10:
                        continue
                        
                    if str(successive_id) in data:
                        successive_ids.append(successive_id)
                        data[str(successive_id)].append(str(cid))
                    else:
                        data[str(successive_id)] = [str(cid)]
            
            for key in list(data.keys()):
                
                if int(key) not in successive_ids:
                    if key in data:
                        data[key].append("None")
                    else:
                        data[key] = ["None"]

    except:
        print("Skipped ",i)
        continue

for key in list(data.keys()):

    if data[key] == ["None"]*132:
        del data[key]
    
for key in list(data.keys()):

    if(len(data[key]) != 132):
        print(key)


id = []
    
for key in list(data.keys()):    

    # Find the majorityID string in data[key]
    majorityID = max((item for item in set(data[str(key)]) if item != "None"), key=data[str(key)].count)
    majorityCount = data[str(key)].count(majorityID)

    patternMajority, patternCount = arrayLongestPattern(data[str(key)])
    
    majorityID = majorityID[2:-1]
    patternMajority = patternMajority[2:-1]

    # if majorityCount > 8:
    id.append((majorityID, majorityCount, patternMajority, patternCount, key))

sorted_list = sorted(id, key=lambda x: (x[3], x[1]), reverse=True)


ids = []
remove = []

id2 = []

for i in sorted_list:
    
    id_to_use = i[0]

    if i[0] != i[2]:
        id_to_use = i[2]
        
    if id_to_use not in ids:
        ids.append(id_to_use)
        id2.append(i)

id2 = id2[:10]
sid = []
ids = {}
priority = {}

for i in id2:
    sid.append(int(i[4]))

for i in id2:
    if i[0] != i[2]:
        ids[int(i[4])] = i[2]
    else:
        ids[int(i[4])] = i[0]

count = 1

for i in id2:
    priority[int(i[4])] = count
    count += 1


frames = {}

for i in range(0,101):
    impath =  f'../data/CH5-png/{i}.png'
    image = cv2.imread(impath)
    count = 0
    frame = []

    for crater in b[i]:
       
        p1 = crater[1][0] * width/100
        p2 = crater[1][1] * height/100
        p3 = crater[1][2] * width/100
        p4 = crater[1][3] * height/100
        p5 = np.radians(crater[1][4])
        suc_id = crater[0]
        param_list = np.array([p1,p2,p3,p4,p5])
        color = (0,0,255)

        if suc_id in sid:
            count += 1
            image = cv2.ellipse(image, (int(p1),int(p2)), (int(p3),int(p4)), int(np.degrees(p5)), 0, 360, color, 2)
            image = cv2.putText(image, str(priority[suc_id]), (int(p1),int(p2)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            frame.append([param_list,ids[suc_id],priority[suc_id]])
    
    if count > 2:
        cv2.imwrite(f'../output/constellation/{i}.png', image)

        sorted_frame = sorted(frame, key=lambda x: x[2])
        frames[i] = sorted_frame[:3]

idx = 101

for i in range(110,420,10):
    impath =  f'../data/CH5-png/{i}.png'
    image = cv2.imread(impath)
    count = 0
    frame = []

    for crater in b[i]:
       
        p1 = crater[1][0] * width/100
        p2 = crater[1][1] * height/100
        p3 = crater[1][2] * width/100
        p4 = crater[1][3] * height/100
        p5 = np.radians(crater[1][4])
        suc_id = crater[0]
        param_list = np.array([p1,p2,p3,p4,p5])
        color = (0,0,255)

        if suc_id in sid:
            count += 1
            image = cv2.ellipse(image, (int(p1),int(p2)), (int(p3),int(p4)), int(np.degrees(p5)), 0, 360, color, 2)
            image = cv2.putText(image, str(priority[suc_id]), (int(p1),int(p2)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            frame.append([param_list,ids[suc_id],priority[suc_id]])
    
    if count > 2:
        cv2.imwrite(f'../output/constellation/{i}.png', image)

        sorted_frame = sorted(frame, key=lambda x: x[2])
        frames[idx] = sorted_frame[:3]
    
    idx += 1


craters_database_text_dir =  '../data/descriptor_db/filtered_catalog_2K.txt' 

CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree = \
        read_crater_database(craters_database_text_dir)

K = get_intrinsic('../data/calibration.pkl')

x = []
y = []
z = []
fnum = []

for i in range(0,132):
    if i in frames:
        frame = frames[i]
        craters_id = [f[1] for f in frame]
        est_cam, cam_in_world_coord = reprojection_test_w_pnp_solver(ID, CW_conic, CW_Hmi_k, CW_params, CW_ENU, K, np.array([f[0] for f in frame]), craters_id)
        x.append(cam_in_world_coord[0])
        y.append(cam_in_world_coord[1])
        z.append(cam_in_world_coord[2])
        fnum.append(i)

# Create a figure
fig = plt.figure()

cmap = ListedColormap(sns.color_palette("crest", 256).as_hex())

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot
scatter = ax.scatter(x, y, z, c=fnum, cmap=cmap, alpha=0.6)

# Add color bar for reference
color_bar = fig.colorbar(scatter, ax=ax)
color_bar.set_ticks([min(fnum),np.median(fnum),max(fnum)])

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# Show the plot
plt.show()
