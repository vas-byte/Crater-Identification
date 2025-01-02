import csv
import numpy as np
import os
import matplotlib.pyplot as plt

def lat_lon_to_cartesian(lat, lon):
    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Convert to Cartesian coordinates (unit vectors)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return x, y, z

def filtering_criterion(filter):
    if filter == 'local':
        diameter_min = 4
        diameter_max = 30
        a_b_ratio = None
        arc_img = 0.9
    elif filter == 'global':
        diameter_min = 25
        diameter_max = 125
        a_b_ratio = 1.1
        arc_img = 0.9
    elif filter == 'config1':
        diameter_min = 3
        diameter_max = 30
        a_b_ratio = None
        arc_img = 0.9
    elif filter == 'config2':
        diameter_min = 3
        diameter_max = 125
        a_b_ratio = None
        arc_img = 0.9
    elif filter == 'config2_local':
        diameter_min = 0
        diameter_max = 30
        a_b_ratio = None
        arc_img = 0.9
    elif filter == 'config2_global':
        diameter_min = 25
        diameter_max = 125
        a_b_ratio = 1.1
        arc_img = 0.9
    elif filter == 'config3':
        diameter_min = 4
        diameter_max = 30
        a_b_ratio = None
        arc_img = 0.85
    elif filter == 'config4':
        diameter_min = 3
        diameter_max = 125
        a_b_ratio = None
        arc_img = 0.85
    elif filter == 'config4_local':
        diameter_min = 3
        diameter_max = 30
        a_b_ratio = None
        arc_img = 0.85
    elif filter == 'config4_global':
        diameter_min = 25
        diameter_max = 125
        a_b_ratio = 1.1
        arc_img = 0.85
    return diameter_min, diameter_max, a_b_ratio, arc_img

if __name__ == "__main__":
    moon_radius = 1737400

    lat_total = []
    long_total = []

    
    csv_dir = 'data/descriptor_db/catalogue.csv'
    output_txt_dir = 'data/descriptor_db/filtered_catalog.txt'

    # Read data from CSV file
    config = 'config2_local'

    with open(csv_dir, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip header
        next(csv_reader)
        with open(output_txt_dir, 'a') as txt_file:
            txt_file.write("selenographic crater coordinates: ID, X, Y, Z, a_dia_metres, b_dia_metres, angle(degree)\n")

        # Open text file for writing
        with open(output_txt_dir, 'a') as txt_file:
            
            # Loop through each row in the CSV
            for row in csv_reader:
                # Extract the first column (ID)
                ID = row[0]
                # Convert the lat and lon to Cartesian coordinates
                try:
                    lat = float(row[3])
                    lon = float(row[4])
                    x, y, z = lat_lon_to_cartesian(lat, lon)
                except:
                    lat = float(row[1])
                    lon = float(row[2])
                    x, y, z = lat_lon_to_cartesian(lat, lon)

                # extract ARC_img
                arc_img = float(row[19])

                # Extract the sixth column
                try:
                    major_len = float(row[7])
                    minor_len = float(row[8])
                    angle = np.deg2rad(float(row[11]))
                except:
                    major_len = float(row[5])
                    minor_len = major_len
                    angle = 0


                a_b_ratio = major_len / minor_len
                diameter = float(row[5])
                diameter_min, diameter_max, a_b_ratio_thres, arc_img_thres = filtering_criterion(config)

                if diameter >= diameter_min and diameter <= diameter_max and arc_img > arc_img_thres and (True if a_b_ratio_thres is None else a_b_ratio <= a_b_ratio_thres):
                    lat_total.append(lat)
                    long_total.append(lon)
                    
                    # Write to text file
                    txt_file.write(f"{ID}, {x * moon_radius:.8f}, {y * moon_radius:.8f}, {z * moon_radius:.8f}, {major_len:.2f}, {minor_len:.2f}, {angle:.2f}\n")

plt.scatter(lat_total, long_total)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Crater Locations')
plt.show()

print("Data processing complete. Check the output.txt file.")
