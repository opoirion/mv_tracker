 # -*- coding: utf-8 -*-

"""config file """

from time import time

import cv2
import numpy as np

###########  Initiation #########
ctime = time()
#################################

######### paramètre de l'algo meanshift ##########
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
##################################################


##### Variable ###########
tseuil = 0.5 # secondes
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


class Config(object):
    """config class """
    def __init__(self,
                 term_crit=term_crit,
                 ctime=ctime,
                 tseuil=tseuil,
                 sum_list_size=sum_list_size,
                 cl_list_size=cl_list_size,
                 kernel=kernel,
                 kernelerode=kernelerode,
                 erodeseuil=erodeseuil,
                 nbproc=nbproc,
                 eps=eps,
                 min_samples=min_samples,
                 reduce_thres=reduce_thres,
                 erode_latence=erode_latence,
                 delete_factor=delete_factor,
                 motion_detection_thres=motion_detection_thres,
    ):
        self.cl_list_size = cl_list_size
        self.term_crit = term_crit
        self.ctime = ctime
        self.tseuil = tseuil
        self.sum_list_size = sum_list_size
        self.kernel = kernel
        self.kernelerode = kernelerode
        self.erodeseuil = erodeseuil
        self.clustermax = clustermax
        self.nbproc = nbproc
        self.eps = eps
        self.min_samples = min_samples
        self.reduce_thres = reduce_thres
        self.erode_latence = erode_latence
        self.delete_factor = delete_factor
        self.motion_detection_thres = motion_detection_thres
