import csv
import pickle
import numpy as np
import cv2

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

parent_dir = '../'

cid_num = {}
num = 0

for i in range(0,101):
    path = parent_dir + f"output/pkl/{i}.pkl"

    try:    
        with open(path, mode ='rb') as file:
            data = pickle.load(file)

            for param_data in data:
                param = param_data[1]
                cid = param_data[0]

                if cid not in cid_num:
                    cid_num[cid] = num
                    num += 1

    except:
        continue


for i in range(110,420,10):
    path = parent_dir + f"output/pkl/{i}.pkl"

    try:    
        with open(path, mode ='rb') as file:
            data = pickle.load(file)

            for param_data in data:
                param = param_data[1]
                cid = param_data[0]
                
                if cid not in cid_num:
                    cid_num[cid] = num
                    num += 1

    except:
        continue


params, image_names = testing_data_read_image_params(parent_dir + 'data/testing_data.csv')

observed = {}

for i in range(len(params)):
    observed[image_names[i]] = []

    for param in params[i]:
        observed[image_names[i]].append([param,None])

for key in observed:
      
    try:
        with open(parent_dir + f'output/pkl/{key}.pkl', 'rb') as file:
            data = pickle.load(file)
        
        for i in range(len(observed[key])):
            for param_data in data:
                param = param_data[1]
                cid = param_data[0]

                if (param == observed[key][i][0]).all():
                    observed[key][i][1] = cid_num[cid]
            
    except:
       continue

for key in observed:

    path = parent_dir + f'data/CH5-png/{key}.png'
    image = cv2.imread(path)

    # iterate through all the parameters and save the images
    for param,id in observed[key]:

        center_coordinates = (param[0], param[1]) 
        axesLength = (param[2] * 2, param[3] * 2)

        angle = np.rad2deg(param[4])

        # Green color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 5 px  
        thickness = 2

        # Using cv2.ellipse() method 
        # Draw a ellipse with red line borders of thickness of 5 px 

        if id != None:
            image = cv2.putText(image, str(id), (int(param[0]), int(param[1] - param[3] - 5)), 2, 0.5, 125)
            color = (0,255,0)

        image = cv2.ellipse(image, (center_coordinates, axesLength, angle), color, thickness)
    
    cv2.imwrite(parent_dir + f'output/images_clear/{key}.png', image)

# Create text file with CID numbers
with open(parent_dir + 'output/images_clear/cid_numbers.txt', 'w') as file:
    for key in cid_num:
        file.write(f'{key}: {cid_num[key]}\n')