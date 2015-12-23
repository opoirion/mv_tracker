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
##### Variable ###########
tseuil = 0.5 # second
kernel = np.ones((2, 2), np.uint8) # Nombre de pixel en x et en y traités par les fonctions filtre: enlève le bruit
kernelerode = np.ones((2, 2), np.uint8) # Nombre de pixel en x et en y traités par les fonctions filtre: enlève le bruit
erodeseuil = 150000 # Somme d'une image max au dela delaquelleest appliqué un filtre d'érosion
nbproc = 1 # Nombre de process effectuant l'algo DBSCAN en parallèle
eps = 50 # paramètre de DBSCAN ~distance min pour un pixel pour intégrer un cluster
min_samples = 60 # paramètre de DBSCAN nombre de pixels min pour former un cluster
clustermax = 25 # Nombre de clusters max en simultané sur une image
cl_list_size = 10 # Inertie de la durée de vie d'une zone vide en mouvement
sum_list_size = 5 # Inertie du seuil de detection d'un mouvement dans l'image
reduce_thres = 25 # seuil (intensité) controlant le déclenchement l'érosion d'une zone de mouvement
erode_latence = 1.0 # latence (s) avant erosion
delete_factor = 0.3 # facteur controlant le déclenchement de la suppression d'une zone de moument
motion_detection_thres = 18000 # seuil de detection d'un mouvement
#########################
```

## test

to launch the algo:
```bash
python mv_tracker/script/main.py
```
