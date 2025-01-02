
# Crater Identification in Chang'e 5 Landing Sequence

This repository contains code to perform Crater Identification on the Chang'e 5 landing sequence using a varation of Christian's method.
The paper describing Christian's method can be accessed via https://link.springer.com/article/10.1007/s40295-021-00287-8

## Setup local envrionment

Clone the project

```bash
  git clone https://github.cs.adelaide.edu.au/a1887068/CraterID
```

Go to the project directory

```bash
  cd christian_cid_method-main
```

Install dependencies

```bash
 pip3 install healpy, numpy, numba, scipy, joblib, matplotlib, opencv-python
```




## Project Structure
**project folder** contains script to run Crater Identification and Build Descriptor Database

&nbsp;

**data folder** contains the data supplied to run the Crater Identification process, along with scripts to prepare the test input and crater catalogue (for descriptor building)

&nbsp;

**visualization folder** contains scripts neccessary to visualize/interpret results

&nbsp;

**output folder**:
- images: outputs for each frame with a match, the labelled crater and ID from Robbin's crater catalogue

- pkl: outputs a pickle file with the parameters of the input crater and predicted ID from Robbin's crater catalogue

In addition, the pnp solver scripts also output

- reprojected: outputs for each frame with a match, the labelled crater and reprojected crater (from catalogue)

- reprojected_pkl: outputs a pickle file with the original labelled parameters, reprojected ellipse parameters, Gaussian Angle, Sigma Square and GA^2/SS^2 distance metric (Note last two are only for ```chris_CID_with_pnp_solver.py``` or ```chris_CID_with_pnp_solver_ga.py```)

&nbsp;

Access to the results is available via the following link https://universityofadelaide.box.com/s/dhn26bvm4jfkigh7wok66b1qrld64fnz

## Usage
Note, all python scripts must be executed in the same directory they are contained in.

### Setting the intrinsic matrix
In the directory data there is a script called calibration.py which creates the intrinsic matrix.

In this case, fx and fy are set by dividing the focal length of the Chang'e 5 camera by the pixel size (both expressed in mm)

w,h represent the width and height of the image frame

cx and cy are the principal points of the chang'e 5 landing camera.

To set the matrix run ```python3 calibration.py``` and set the above parameters as needed

##
### Building the crater catalogue.

Download the robbins crater database from https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins and place the unzipped folder in data/descriptor_db/

&nbsp;

Then run ```python3 subset_db.py``` to get a subset of the catalogue. Filtering is performed using lattitude and longitude. For Chang'e 5, craters between 285° and 320° longitude and those between 30° and 50° latitude were included. This script produces a csv file named catalogue.csv

&nbsp;

Then run ```python3 craters_database_filtering.py``` which produces a text file further filtering the ellipse objects describing the craters (by size and arc completness). This script also converts the lattitude and longitude describing each ellipse into parameters [x,y,a,b,phi], which are expressed as cartesian coordinates.  

&nbsp;

Then run ```python3 descriptor_building.py``` to build the catalogue. An nside value of 32 (healpix) was used for testing.

&nbsp;

Then run ```python3 db_compilation.py``` to build the catalogue.

&nbsp;

Alternatively, you may download the catalogue covering for the Chang'e 5 landing sequence from https://universityofadelaide.box.com/s/pbqtjhevivhe74wkl402typk42e5h330

##
### Preparing Test Input
In the Chang'e 5 Dataset by Mathhew Roddha, a joblib file named ```CE5-ellipse-labels``` is provided, this must be converted into a csv. This can be done by running the python script below:
```python3 create_test_input.py```

This dataset is accessible at https://zenodo.org/records/11326450

##
### Running the CID Methods
Create a folder called images and pkl in the output directory

Chirstian's method without using the reprojection test verification, run 

```python3 chris_CID.py```

&nbsp;

Create foldrs: images, pkl, reprojected and reprojected_pkl in the output directory.

To run Christian's method using the reprojection test and Gausian Angle distance metric

```python3 chris_CID_with_pnp_solver.py```

Note for this script, we used a matching_percentage of 0.5 and img_sigma of 3.

&nbsp;

Create foldrs: images, pkl, reprojected and reprojected_pkl in the output directory.

To run Christian's method using reprojection and raw Gausian Angle

```python3 chris_CID_with_pnp_solver_ga.py```

Note matching_percentage of 0.5 was used, raw Gaussian Angle of 2.1 was used as a threshold to confirm a match.

##
### Scripts to interpret results

Note, the terminology "tracked crater" refers to an observed crater that is visible (and labelled) over multiple frames of the Chang'e 5 landing sequence. 


```cofident_id_distribution.py``` shows the distribtuion of IDs from the robbins crater catalogue across the confidently predicted subset of tracked craters from the Roddha Dataset. A confident prediction occurs when the majority ID predicted for a tracked crater appears in more than half of the labeled frames, and the tracked crater is labelled for ten or more frames.

&nbsp;

```consistency_ratio_plot.py``` plots a bar chart showing the ratio of frames where the majority ID is predicted to the total number of labeled frames. Only tracked craters with confident, unique majority ID predictions are used (the vote-and-identify method can match multiple observed craters in the same frame to a single ID in the Robbins Database).

&nbsp;

```crater_freq_frames.py``` visualizes the frequency of the majority ID up to the current frame in the landing sequence. It provides a good idea of how frequently the majority ID is predicted relative to other craters in the sequence.

&nbsp;

```crater_freq_plot.py``` generates a bar chart showing the frequency of the majority ID for tracked craters. This script specifically visualizes results for the PnP method using the raw GA metric, aiming to illustrate matching stability across the sequence. It plots the frequency of majority ID predictions for frames 270–330 or 380–410.

&nbsp;

``crater_id_shortened.py`` produces a visual showing each labelled crater and a shortened associtaed ID from the robbins catalogue (the ID is a number starting from 1). A text file is used to highlgiht the mapping of the shortened ID (1 to ...) to the catalogue ID. This declutters the visualization as the RAW IDs can overlap in dense regions of labels.

&nbsp;

``ga_experiment.py`` produces a visual showing the raw Gausian Angle, sigma square and Gaussian Angle distance metric for a large and small ellipse observation. Then by simulating a close and distant ellipse reprojection, the script can be used to observe how the Gaussian Angle and Gaussian Angle distance metric change.

&nbsp;

```ga_experiment_c.py ``` an experiment to show how the scalar c changes with differing ellipse parameters in the calculation of raw Gaussian Angle.

&nbsp;

``ga_experiment_raw.py`` pertubes a crater ellipse and outputs the relevant Gaussian Angle. This was used to find the threshold for the ``chris_CID_with_pnp_solver_ga.py`` which uses a raw Gaussian Angle matching threshold.

&nbsp;

```python3 projection_constellation.py ``` outputs a plot showing estimated trajectory of CE5 using estimated camera positions (SQPnP solver) from a subset of high quality crater matches identified by the script ```chris_CID.py```
