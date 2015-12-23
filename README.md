# Movement tracker using python with dbscan and meanshift algo

some old toy works about movement detection:

building a movement tracker using python with opencv and sklearn libs.
The tracker is then used on the webcam.

## Installation

* dependencies
 * scikit-learn
 * numpy
 * opencv (for c++) and cv2 (opencv wrapper) for python
* to install:
```bash
pip install -U -r requirements.txt
```

## principle
* movement is computed by images difference
* a noise reduction function is then applied by eroding the movement image proportionnal to the intensity
* a threshold controls the triggering of the clustering procedure
* a dbscan clustering procedure is performed to detect movement pixel clusters
* meanshift trackers are then applied to each movement areas
* processing of the different movement area are then effectued (deletion, erosion, fusion) according to different parameters
*
# config
All the parameters controling the algorithm are defined into the `config.py` file:


```python
#########  meanshift parameter ######## ##########
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
##################################################


##### Variable ###########
tseuil = 0.5 # seconds
kernel = np.ones((2, 2), np.uint8) # Noise dealer: Number of pixel in x and y removed
kernelerode = np.ones((2, 2), np.uint8) # Number of pixel in x and y removed for each erode iter
erodeseuil = 150000 # erosion process threshold triggering
nbproc = 1 # nb of clustering processes to launch in parallel
eps = 50 # DBSCAN parameter ~min distance for a pixel to be allowed to integrate a cluster
min_samples = 60 # DBSCAN parameter ~ min number of pixels to form a cluster
clustermax = 25 # max number of simultaned clusters
cl_list_size = 10 # lifetime inertia of a movement area
sum_list_size = 5 # Inertia of the movement detection threshold
reduce_thres = 25 # threshold (pixels sum intensity) triggering area erosion
erode_latence = 1.0 # latence (s) before erosion
delete_factor = 0.3 # factor controling the suppresion of a movement area
motion_detection_thres = 18000 # threshold for movement detection
#########################
```

## test

to launch the algo:
```bash
python mv_tracker/script/main.py
```
