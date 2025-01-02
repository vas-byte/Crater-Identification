import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

id = {}

height = 1728
width = 2352

# Get directory of pickle file
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

for i in range(0,101):

    path = f"../output/pkl/{i}.pkl"

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

                        param_list = [p1,p2,p3,p4,p5]

                        if (param_list == param).all():
                            successive_id = crater[0]
                            break

                    if successive_id == -1:
                        continue

                    if SuccessiveCount[successive_id] < 10:
                        continue
                    
                    if successive_id in id:
                        
                        if cid in id[successive_id]:
                            id[successive_id][cid] += 1

                        else:
                            id[successive_id][cid] = 1

                    else:
                        id[successive_id] = {}
                        id[successive_id][cid] = 1

    except Exception as e:
        print(e)
        continue

for i in range(110,420,10):

    path = f"../output/pkl/{i}.pkl"

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

                        param_list = [p1,p2,p3,p4,p5]

                        if (param_list == param).all():
                            successive_id = crater[0]
                            break

                    if successive_id == -1:
                        continue

                    if SuccessiveCount[successive_id] < 10:
                        continue
                    
                    if successive_id in id:
                        
                        if cid in id[successive_id]:
                            id[successive_id][cid] += 1

                        else:
                            id[successive_id][cid] = 1

                    else:
                        id[successive_id] = {}
                        id[successive_id][cid] = 1
    except:
        continue

crater_id = {}
count_id = {}
consistency_ratio = {}

# Check for threshold where ID is consistent for half (or more) successive craters
for i in range(0,1337):

    shouldInclude = False

    # get the id's for each successive crater
    if i in id:
        ids = list(id[i].keys())

        maxCID = ""
        maxCount = 0

        for cid in ids:
            if id[i][cid] > maxCount:
                maxCount = id[i][cid]
                maxCID = cid
        
        if id[i][maxCID] > (SuccessiveCount[i]/2):
            if maxCID in crater_id:
                crater_id[maxCID] += 1
            else:
                crater_id[maxCID] = 1
        
for i in range(0,1337):

    shouldInclude = False

    # get the id's for each successive crater
    if i in id:
        ids = list(id[i].keys())

        maxCID = ""
        maxCount = 0

        for cid in ids:
            if id[i][cid] > maxCount:
                maxCount = id[i][cid]
                maxCID = cid
        
        if maxCID in crater_id and crater_id[maxCID] == 1:
            if maxCID in count_id:
                if count_id[maxCID][0] < maxCount:
                    count_id[maxCID] = (maxCount,i)     
            else:
                count_id[maxCID] = (maxCount,i)  

for key in count_id.keys():
    consistency_ratio[key] = count_id[key][0]/SuccessiveCount[count_id[key][1]]

# Data
categories = consistency_ratio.keys()
values = consistency_ratio.values()

# Create the bar chart
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Robbins Crater ID')
plt.ylabel('Consistency Ratio')
plt.title(f'Consistency Ratios of Confident and Unique Tracked Crater Matches')

# Display the plot
plt.show()
