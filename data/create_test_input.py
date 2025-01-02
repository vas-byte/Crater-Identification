import joblib
import sys
import numpy
import csv
import numpy as np

loaded = joblib.load("CE5-ellipse-labels")

data = []


for i in loaded:

    ellipse_sparse = []

    for matrix in i['ellipse_sparse']:
        ellipse_sparse.append([matrix[0]*2352/100,matrix[1]*1728/100,matrix[2]*2352/100,matrix[3]*1728/100,float(np.radians(matrix[4]))])

    data.append({'Camera Extrinsic': None, 
                 'Camera Pointing Angle': None,
                 'Imaged ellipses': ellipse_sparse,
                 'Imaged ellipses with noise': ellipse_sparse,
                 'Crater Indices': None,
                 'Height': None,
                 'Noise level': None,
                 'Remove_percentage': None,
                 'Add_percentage': None,
                 'Att_noise': None,
                 'Noisy cam orientation': None,
                 'image': int(i['id'][:-4])
                 })

data = sorted(data, key=lambda d: d['image'])

with open('testing_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Camera Extrinsic', 'Camera Pointing Angle', 'Imaged ellipses', 'Imaged ellipses with noise', 'Crater Indices', 'Height', 'Noise level', 'Remove_percentage', 'Add_percentage', 'Att_noise', 'Noisy cam orientation', 'image']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

