# Motion tracker using python with dbscan and meanshift algo

some old works about movement detection I did a while ago!:

building a movement tracker using python with opencv and sklearn libs.
The tracker is then used on the webcam.

## Installation

Tested with python2. Requires the cv2 wrapper for python2 (I personnally installed cv2wrap)

* dependencies
 * scikit-learn
 * numpy
 * scipy
 * opencv (for c++) and python-opencv (opencv wrapper) for python
* to install:
```bash
pip install -r requirements.txt --user
```

However, sometime the installation through pip is buggy, depending of the distribution. Also, the cv2 package to install depends of the OS. (I'm not sure that cv2wrap will be ok in any situations). A solution is to install manually the different package using the package manager of the distribution (i.e. `apt-get install python-opencv`...etc... and in a second time to install the package without dependency.

```bash
pip install --no-deps --user
```

Once mv_tracker is installed, a webcam needs to be plugged. cv2 should also be able to read an image from the webcam

* test the package

```bash
pip install nose --user
nosetests -v
```

```python
# test cv2 in python2
import cv2

cam  =  cv2.VideoCapture(0)


winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

img = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
print img

while True:
      img = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
      cv2.imshow(winName, img)

```


## principle
* movement is computed by images difference
* a noise reduction function is then applied by eroding the movement image proportionnal to the intensity
* a threshold controls the triggering of the clustering procedure
* a dbscan clustering procedure is performed to detect movement pixel clusters
* meanshift trackers are then applied to each movement areas
* processing of the different movement area are then effectued (deletion, erosion, fusion) according to different parameters

## config
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
eps = 50 # DBSCAN parameter ~min distance for a pixel to be allowed to integrate a cluster, control the "granularity" of the tracking area
min_samples = 60 # DBSCAN parameter ~ min number of pixels to form a cluster, prevent noisy clusters
clustermax = 25 # max number of simultaned clusters
cl_list_size = 10 # lifetime inertia of a movement area
sum_list_size = 5 # Inertia of the movement detection threshold
reduce_thres = 25 # threshold (pixels sum intensity) triggering area erosion
erode_latence = 1.0 # latence (s) before erosion
delete_factor = 0.3 # factor controling the suppresion of a movement area
motion_detection_thres = 18000 # threshold for movement detection
#########################
```

## interest
the different parameters could (in theory!) allowing us to control perfectly the type of movement to track (small objects very noisy, big objects slowly moving...)

## test

to launch the algo:
```bash
python mv_tracker/script/main.py
```

## demo

![video face](./demo/demo_face.gif)
![video person](./demo/demo_person.gif)

## To do!

* add video reader (only from webcam currently!) [easy]
* add supervized analysis to detect head, hand, person...etc... [long but largely doeable]
* test other tracking algo (i.e. camshift) [don't know if it will work]
* allow rectangle rotation (camshift can answer to that else use of SVG matrix decomposition) [long, could be hard]
* add "anchor points" to keep tracking newly static objcts [hard]
* optimize speed and memory (python!! and quickly done), cython could be used for some parts. else recoding all the project in C++ instead (under the condition to find a fitable dbscan algo)
