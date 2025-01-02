import joblib
import sys
import numpy
import csv
import json

base_path = "/Users/vasilismichalakis/Documents/Uni/2nd Year/Sem 2/Topics/Code.nosync/christian_cid_method-main/data/"
path = base_path + "CE5-ellipse-labels"

loaded = joblib.load(path)

for i in loaded:

    # Write to a text file
    with open(f"Original Params/{i['id']}.json", 'w') as f:
        json.dump(i['ellipse_sparse'], f, indent=1)  # Use indent for pretty formatting