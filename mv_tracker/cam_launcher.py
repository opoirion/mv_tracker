 # -*- coding: utf-8 -*-

"""class processing the cam """

from mv_tracker.processes.dbscan_clustering import ClusteringProcess
from mv_tracker.config import Config

from multiprocessing import Queue

import cv2

import numpy as np
from time import time

from collections import defaultdict


class CamLauncher(object):

    def __init__(self, config=Config()):
        self.config = config
        self.recdic = None
        self.recdata = None
        self.imglist = None
        self.outqueue = None
        self.count = None
        self.debug = False
        self.processlist = []
        self.img = None

        self.img_mvt = None
        self.img = None
        self.img_plus = None
        self.img_minus = None

    def _init_var(self):
        self.recdic = {}
        self.recdata = {}
        self.imglist = Queue()
        self.outqueue = Queue()
        self.count = 0
        self.sumlist = [0 for i in
                        range(self.config.sum_list_size)]

        #Lancement des process de DBSCAN
        for p in self.processlist:
            p.terminate()
        self.processlist = [
            ClusteringProcess(i,
                              1,
                              self.imglist,
                              self.outqueue,
                              self.config)
            for i in range(self.config.nbproc)]
        for p in self.processlist:
            p.start()

    def _measure_diff_img(self):
        """ """
        a = diffImg(
            self.img_minus,
            self.img,
            self.img_plus)

        ret, self.img_mvt = cv2.threshold(
            a, 15, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        self.img_mvt = cv2.morphologyEx(
            self.img_mvt, cv2.MORPH_OPEN, self.config.kernel)
        Sum =  self.img_mvt.sum()
        self.sumlist.append(Sum)
        self.sumlist.pop(0)

    def _rectangle_processing(self):
        """process tracking area (earasing, fusioning, reducing..) """
        rmkey = []

        for key in self.recdic.keys():
            x, y, w, h = self.recdic[key]
            a_square = self.img_mvt.T[x:x+w, y+h: y]
            cl_list = self.config.cl_list_size

            self.recdata[key]['square'] = (cl_list - 1.0)/ cl_list * self.recdata[key]['square'] \
                                          + 1.0 / cl_list * a_square

            self.recdata[key]['sum'].append(np.sum(self.img_mvt.T[x:x+w, y+h: y]))
            self.recdata[key]['sum'].pop(0)
            meankey = np.mean(self.recdata[key]['sum'])

            if meankey < self.config.delete_factor * self.recdata[key]['mean']:
                # on supprime le rectangle
                rmkey.append(key)

            # reduit la taille tu rectangle selon les extremités
            if self.recdata[key]['iter'] > self.config.erode_latence / self.config.tseuil:
                n_square = self.recdata[key]['square'][:]
                n_square[n_square < self.config.reduce_thres] = 0
                M = np.transpose(np.nonzero(n_square.T))
                if M.any():
                    amin, amax, omin, omax = (M.T[0].min(),
                                              M.T[0].max(),
                                              M.T[1].min(),
                                              M.T[1].max())

                    x, y, w, h = (omin + x,
                                  amin + y,
                                  omax - omin + 1,
                                  amin - amax - 1)
                    # import ipdb;ipdb.set_trace()
                    self.recdic[key] = x, y, w, h
                    self.recdata[key]['square'] = self.img_mvt.T[x:x+w, y+h: y]\
                                                              .astype('float32')
                    self.recdata[key]['iter'] = 0
            else:
                self.recdata[key]['iter'] += 1
        for key in rmkey:
            self.recdata.pop(key)
            self.recdic.pop(key)

        self._fusion()

    def _apply_tracker(self, roi_hist):
        ret, self.img_mvt = cv2.threshold(
            self.img_mvt, 0, 255, cv2.THRESH_BINARY_INV)
        self.img_mvt = cv2.cvtColor(self.img_mvt, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(self.img_mvt, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv], [2], roi_hist, [0, 255], 1)

        for key in self.recdic:
            track_window = (self.recdic[key][0],
                            self.recdic[key][1] + self.recdic[key][3],
                            self.recdic[key][2],
                            abs(self.recdic[key][3]))
            x, y, w, h = track_window
            if w * h:
                ret, track_window = cv2.meanShift(dst,
                                                  track_window,
                                                  self.config.term_crit)
            x, y, w, h = track_window
            cv2.rectangle(self.img_mvt, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.recdic[key] = x, y+h, w, -h
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def _get_clusters(self):
        """ """
        try:
            shapes = self.outqueue.get(True, 0.2)
        except Exception as e:
            print e
        else:
            for shape in shapes:
                self.count += 1
                x, y, w, h = shape[0]
                self.recdata[self.count] = defaultdict(list)
                self.recdata[self.count]['square'] = self.img_mvt.T[x:x+w, y+h: y]\
                                                                      .astype('float32')
                self.recdata[self.count]['iter'] = 0

                for i in range(self.config.cl_list_size):
                    self.recdata[self.count]['sum'].append(shape[1])

                self.recdata[self.count]['mean'] = shape[1]
                self.recdic[self.count] = shape[0]

    def _fusion(self):
        """ fonction de fusion des rectangles """
        keys = self.recdic.keys()

        for i in range(len(keys)):

            if keys[i] not in self.recdic:
                continue
            for j in range(i):
                if keys[j] not in self.recdic:
                    continue

                x1, y2, w1, h1 = self.recdic[keys[i]]
                y1, x2 = y2 + h1, x1 + w1
                xa, yb, wa, ha = self.recdic[keys[j]]
                ya, xb = yb + ha, xa + wa
                var = False
                if x1 < xa < x2 and (y1 < ya <y2 or y1 < yb < y2):
                    var = True
                elif x1< xb < x2 and (y1 <ya < y2 or y1 < yb < y2):
                    var = True
                elif xa < x1 < xb and (ya < y1 < yb or ya < y2 < yb):
                    var = True
                elif xa < x2 < xb and (ya < y1 < yb or ya < y2 < yb):
                    var = True

                if not var:
                    continue

                A1 = w1 * h1
                Aa = wa * ha
                xl = np.sort([x1, x2, xa, xb])
                yl = np.sort([y1, y2, ya, yb])
                As = (xl[2] - xl[1]) * (yl[2] - yl[1])

                if As > min([A1, Aa]):

                    if min([A1 ,Aa]) == A1:
                        i, j, A1, Aa = j, i, Aa, A1
                        self.recdic[keys[i]] = xa, yb, wa, ha

                    self.recdata[keys[i]]['mean'] = (A1 * self.recdata[keys[i]]['mean'] \
                                                + self.recdata[keys[j]]['mean'] * Aa) / (A1 + Aa)
                    self.recdata[keys[i]]['sum'] = ((A1 * np.asarray(self.recdata[keys[i]]['sum']) \
                                                + np.asarray(self.recdata[keys[j]]['sum']) * Aa) / \
                                               (A1 + Aa)).tolist()
                    (x, y, w, h) = (min(x1, xa),
                                    max(y2, yb),
                                    max(x2, xb)-min(x1,xa),
                                    min(y1,ya) - max(y2,yb))

                    if (x, y, w, h) != self.recdic[keys[i]]:
                        self.recdata[keys[i]]['square'] = self.img_mvt.T[x:x+w, y+h: y]\
                                                                      .astype('float32')
                        self.recdic[keys[i]] = (x, y, w, h)

                    self.recdic.pop(keys[j])
                    self.recdata.pop(keys[j])

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt as e:
            print 'exception:{0}\nterminating...'.format(e)
            for p in self.processlist:
                p.terminate()
            return

    def _run(self):

        self._init_var()

        cam  =  cv2.VideoCapture(0)
        s, img  =  cam.read()

        #############################################################
        winName = "Movement Indicator"
        cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)
        winName2 = "Movement Indicator 2"
        cv2.namedWindow(winName2, cv2.CV_WINDOW_AUTOSIZE)
        self.img_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        self.img = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        self.img_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        #############################################################
        # Le mouvement est detecté par la diff de deux images successives
        a = diffImg(self.img_minus,
                    self.img,
                    self.img_plus)
        ################## Initiation MeanShift #####################
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, self.config.kernel) # filtre
        ret,a3 = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV) # filtre meanshift
        a3 = cv2.cvtColor(a3, cv2.COLOR_GRAY2RGB)
        hsv_roi =  cv2.cvtColor(a3, cv2.COLOR_RGB2HSV)
        mask = None
        roi_hist = cv2.calcHist([hsv_roi], [2], mask, [255], [0, 255])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        r,h,c,w = 150, 80, 230, 20
        track_window = (c, r, w, h)
        #############################################################

        var = True

        while s:
            self._measure_diff_img()

            for p in self.processlist:
                if not p.is_alive():
                    print 'probleme process'
                    raise KeyboardInterrupt

                if time() - self.config.ctime > self.config.tseuil:
                    var = True
                if np.mean(self.sumlist) > self.config.motion_detection_thres \
                   and var and \
                   ((self.config.clustermax and \
                     len(self.recdic) < self.config.clustermax)
                    or not self.config.clustermax):
                        if sum([p.status for p in self.processlist]):
                            # Si les conditions sont respectés
                            # envoie aux process une image à clusteriser
                            self.imglist.put(self.img_mvt)
                            var = False
                            self.config.ctime = time()
                if not self.outqueue.empty():
                    #recuperation des clusters
                    self._get_clusters()
                if self.recdic:
                    # si des rectangles existent:
                    # traitement (agrandissement, supression, retrecissement)
                    self._rectangle_processing()

            if self.recdic:
                # si il existe des rectangles (ou zones d'interet)
                # on applique meanshift sur ceux-ci
                t = self._apply_tracker(roi_hist)

            # Affichage du mouvement et de l'image normale
            cv2.imshow(winName, self.img)
            cv2.imshow(winName2, self.img_mvt)

            self.img_minus = self.img
            self.img = self.img_plus
            self.img_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

            key = cv2.waitKey(20)

            if key  ==  27:
                cv2.destroyWindow(winName)
                break
                print "Goodbye"


def diffImg(t0, t1, t2):
  """ difference de deux images """
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)
