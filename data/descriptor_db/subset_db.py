import csv
import os

data = []
path = 'lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv'

# For larger area I'm using lon 285 to 320 and lat 30 to 50
# For small patch, I' using lon 280 to 310 and lat 35 to 45

with open(path, mode ='r') as file:    
       csvFile = csv.DictReader(file)
       for lines in csvFile: 
            try:
                if float(lines['LON_ELLI_IMG']) > 280 and float(lines['LON_ELLI_IMG']) < 310 and float(lines['LAT_ELLI_IMG']) > 35 and float(lines['LAT_ELLI_IMG']) < 45 :
                  data.append(lines)
            except:
                 continue

with open('catalogue.csv' , 'w', newline="") as employee_file:
     fieldnames2 = ['CRATER_ID' , 'LAT_CIRC_IMG' , 'LON_CIRC_IMG' , 'LAT_ELLI_IMG' , 'LON_ELLI_IMG' , 'DIAM_CIRC_IMG', 'DIAM_CIRC_SD_IMG', 'DIAM_ELLI_MAJOR_IMG', 'DIAM_ELLI_MINOR_IMG', 'DIAM_ELLI_ECCEN_IMG', 'DIAM_ELLI_ELLIP_IMG', 'DIAM_ELLI_ANGLE_IMG', 'LAT_ELLI_SD_IMG', 'LON_ELLI_SD_IMG', 'DIAM_ELLI_MAJOR_SD_IMG', 'DIAM_ELLI_MINOR_SD_IMG', 'DIAM_ELLI_ANGLE_SD_IMG', 'DIAM_ELLI_ECCEN_SD_IMG', 'DIAM_ELLI_ELLIP_SD_IMG', 'ARC_IMG', 'PTS_RIM_IMG']
     csvwriter = csv.writer(employee_file , delimiter = ',')
     csvwriter.writerow(fieldnames2)
     for row in data:
        csvwriter.writerow(row.values())
