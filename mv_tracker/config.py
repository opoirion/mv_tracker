 # -*- coding: utf-8 -*-

"""config file """

from time import time

import cv2
import numpy as np

###########  Initiation #########
ctime = time()
#################################

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
