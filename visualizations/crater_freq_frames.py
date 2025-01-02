import pickle
import numpy as np
import cv2
import csv

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

# key: successive crater id, value: dictionary of key: crater id, value: count
id = {}

height = 1728
width = 2352

# Get directory of pickle file
successive_path = "../data/Successive Craters/params.pickle"

with open(successive_path, 'rb') as handle:
    b = pickle.load(handle)

testing_params, _ = testing_data_read_image_params('../data/testing_data.csv')

for i in range(0,101):

    path = f"../output/pkl/{i}.pkl"
    impath =  f'../data/CH5-png/{i}.png'
    image = cv2.imread(impath)

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
                        
                        for j in range(len(testing_params[i])):
                            if (testing_params[i][j] == param).all():
                                testing_params[i].pop(j)
                                break

                        break

                if successive_id == -1:
                    continue

                
                if successive_id in id:
                    
                    if cid in id[successive_id]:
                        id[successive_id][cid] += 1

                    else:
                        id[successive_id][cid] = 1

                else:
                    id[successive_id] = {}
                    id[successive_id][cid] = 1
                
                maxID = 0

                for cid in id[successive_id]:
                    maxID = max(maxID, id[successive_id][cid])
                
                center_coordinates = (param[0], param[1]) 
                axesLength = (param[2] * 2, param[3] * 2)

                angle = np.rad2deg(param[4])

                # Green color in BGR 
                color = (0, 255, 0) 

                # Line thickness of 5 px  
                thickness = 2

                # Using cv2.ellipse() method 
                # Draw a ellipse with red line borders of thickness of 5 px 
                image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)
                image = cv2.putText(image, f"{maxID}", (int(param[0]), int(param[1] - param[3] - 5)), 2, 1, 125,3)
            
            for rem in testing_params[i]:
                
                maxID = 0

                for crater in b[i]:
                
                    p1 = crater[1][0] * width/100
                    p2 = crater[1][1] * height/100
                    p3 = crater[1][2] * width/100
                    p4 = crater[1][3] * height/100
                    p5 = np.radians(crater[1][4])

                    param_list = np.array([p1,p2,p3,p4,p5])

                    if (param_list == rem).all():
                        successive_id = crater[0]
                       
                if successive_id in id:
                    for cid in id[successive_id]:
                        maxID = max(maxID, id[successive_id][cid])
                    
                center_coordinates = (rem[0], rem[1]) 
                axesLength = (rem[2] * 2, rem[3] * 2)

                angle = np.rad2deg(rem[4])

                # Red color in BGR 
                color = (0, 0, 255) 

                # Line thickness of 5 px  
                thickness = 2

                # Using cv2.ellipse() method 
                # Draw a ellipse with red line borders of thickness of 5 px 
                image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)
                image = cv2.putText(image, f"{maxID}", (int(center_coordinates[0]), int(center_coordinates[1] - axesLength[1] - 5)), 2, 1, 125,3)

            cv2.imwrite(f"../output/images_freq/{i}.png", image)

    except:
        print("Skipped ",i)

        for rem in testing_params[i]:
                    
            center_coordinates = (rem[0], rem[1]) 
            axesLength = (rem[2] * 2, rem[3] * 2)

            angle = np.rad2deg(rem[4])

            # Red color in BGR 
            color = (0, 0, 255) 

            # Line thickness of 5 px  
            thickness = 2

            # Using cv2.ellipse() method 
            # Draw a ellipse with red line borders of thickness of 5 px 
            image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)

        cv2.imwrite(f"../output/images_freq/{i}.png", image)

        continue

index = 101

for i in range(110,420,10):

    path = f"../output/pkl/{i}.pkl"
    impath =  f'../data/CH5-png/{i}.png'
    image = cv2.imread(impath)

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
                        
                        for j in range(len(testing_params[index])):
                            if (testing_params[index][j] == param).all():
                                testing_params[index].pop(j)
                                break

                        break

                if successive_id == -1:
                    continue

                    
                if successive_id in id:
                    
                    if cid in id[successive_id]:
                        id[successive_id][cid] += 1

                    else:
                        id[successive_id][cid] = 1

                else:
                    id[successive_id] = {}
                    id[successive_id][cid] = 1
                
                maxID = 0

                for cid in id[successive_id]:
                    maxID = max(maxID, id[successive_id][cid])
                
                center_coordinates = (param[0], param[1]) 
                axesLength = (param[2] * 2, param[3] * 2)

                angle = np.rad2deg(param[4])

                # Green color in BGR 
                color = (0, 255, 0) 

                # Line thickness of 5 px  
                thickness = 2

                # Using cv2.ellipse() method 
                # Draw a ellipse with red line borders of thickness of 5 px 
                image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)
                image = cv2.putText(image, f"{maxID}", (int(param[0]), int(param[1] - param[3] - 5)), 2, 1, 125,3)

            for rem in testing_params[index]:

                maxID = 0

                for crater in b[i]:
                
                    p1 = crater[1][0] * width/100
                    p2 = crater[1][1] * height/100
                    p3 = crater[1][2] * width/100
                    p4 = crater[1][3] * height/100
                    p5 = np.radians(crater[1][4])

                    param_list = np.array([p1,p2,p3,p4,p5])

                    if (param_list == rem).all():
                        successive_id = crater[0]
                       
                if successive_id in id:
                    for cid in id[successive_id]:
                        maxID = max(maxID, id[successive_id][cid])
                    
                center_coordinates = (rem[0], rem[1]) 
                axesLength = (rem[2] * 2, rem[3] * 2)

                angle = np.rad2deg(rem[4])

                # Red color in BGR 
                color = (0, 0, 255) 

                # Line thickness of 5 px  
                thickness = 2

                # Using cv2.ellipse() method 
                # Draw a ellipse with red line borders of thickness of 5 px 
                image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)
                image = cv2.putText(image, f"{maxID}", (int(center_coordinates[0]), int(center_coordinates[1] - axesLength[1] - 5)), 2, 1, 125,3)

            cv2.imwrite(f"../output/images_freq/{i}.png", image)
            index += 1

    except:
        print("Skipped ",i)

        for rem in testing_params[index]:
                    
            center_coordinates = (rem[0], rem[1]) 
            axesLength = (rem[2] * 2, rem[3] * 2)

            angle = np.rad2deg(rem[4])

            # Red color in BGR 
            color = (0, 0, 255) 

            # Line thickness of 5 px  
            thickness = 2

            # Using cv2.ellipse() method 
            # Draw a ellipse with red line borders of thickness of 5 px 
            image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)

        cv2.imwrite(f"../output/images_freq/{i}.png", image)
        index += 1
        continue
